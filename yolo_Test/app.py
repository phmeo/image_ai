import os
import io
import time
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, render_template, send_from_directory, url_for

from detector import YOLODetector


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
DB_PATH = os.path.join(BASE_DIR, "history.db")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


app = Flask(
    __name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR
)

detector = YOLODetector(model_name=os.environ.get("YOLO_MODEL", "yolov8n.pt"))


# ---------- Database utilities ----------


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_filename TEXT NOT NULL,
                source_type TEXT NOT NULL,
                output_relpath TEXT NOT NULL,
                classes_json TEXT NOT NULL,
                confs_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                model TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                conf REAL NOT NULL,
                iou REAL NOT NULL
            )
            """
        )
        conn.commit()


_init_db()


def _insert_history(
    source_filename: str,
    source_type: str,
    output_relpath: str,
    classes: List[str],
    confs: List[float],
    model: str,
    duration_ms: int,
    conf: float,
    iou: float,
) -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO detections (
                source_filename, source_type, output_relpath, classes_json, confs_json,
                created_at, model, duration_ms, conf, iou
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_filename,
                source_type,
                output_relpath,
                json.dumps(classes),
                json.dumps(confs),
                datetime.utcnow().isoformat(timespec="seconds") + "Z",
                model,
                duration_ms,
                conf,
                iou,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


# ---------- Helpers ----------


def _allowed_file(filename: str) -> Tuple[bool, str]:
    lower = filename.lower()
    image_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    video_exts = (".mp4", ".mov", ".avi", ".mkv", ".webm")
    if lower.endswith(image_exts):
        return True, "image"
    if lower.endswith(video_exts):
        return True, "video"
    return False, ""


# ---------- Routes ----------


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/detect")
def api_detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    ok, source_type = _allowed_file(file.filename)
    if not ok:
        return jsonify({"error": "Unsupported file type"}), 400

    try:
        conf = float(request.form.get("conf", 0.35))
        iou = float(request.form.get("iou", 0.45))
    except Exception:
        conf = 0.35
        iou = 0.45

    # Save upload
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    safe_name = f"{timestamp}_{os.path.basename(file.filename)}"
    src_path = os.path.join(UPLOADS_DIR, safe_name)
    file.save(src_path)

    # Run detection
    t0 = time.time()
    try:
        det = detector.predict(src_path, OUTPUTS_DIR, conf=conf, iou=iou)
    except Exception as e:
        return jsonify({"error": f"Detection failed: {e}"}), 500
    duration_ms = int((time.time() - t0) * 1000)

    # Store history
    rel = os.path.relpath(det["output_path"], OUTPUTS_DIR)
    history_id = _insert_history(
        source_filename=safe_name,
        source_type=source_type,
        output_relpath=rel.replace("\\", "/"),
        classes=det.get("classes", []),
        confs=[float(x) for x in det.get("confs", [])],
        model=det.get("model", "unknown"),
        duration_ms=duration_ms,
        conf=conf,
        iou=iou,
    )

    return jsonify(
        {
            "id": history_id,
            "output_url": url_for("serve_output", filename=rel, _external=False),
            "classes": det.get("classes", []),
            "confs": [float(x) for x in det.get("confs", [])],
            "model": det.get("model", "unknown"),
            "duration_ms": duration_ms,
            "source_type": source_type,
        }
    )


@app.get("/api/history")
def api_history():
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT id, source_filename, source_type, output_relpath, created_at, model, duration_ms, conf, iou FROM detections ORDER BY id DESC"
        ).fetchall()
    items = []
    for r in rows:
        rel = r["output_relpath"]
        items.append(
            {
                "id": int(r["id"]),
                "source_filename": r["source_filename"],
                "source_type": r["source_type"],
                "output_url": url_for("serve_output", filename=rel, _external=False),
                "created_at": r["created_at"],
                "model": r["model"],
                "duration_ms": int(r["duration_ms"]),
                "conf": float(r["conf"]),
                "iou": float(r["iou"]),
            }
        )
    return jsonify(items)


@app.get("/api/history/<int:det_id>")
def api_history_item(det_id: int):
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM detections WHERE id = ?",
            (det_id,),
        ).fetchone()
    if row is None:
        return jsonify({"error": "Not found"}), 404

    rel = row["output_relpath"]
    return jsonify(
        {
            "id": int(row["id"]),
            "source_filename": row["source_filename"],
            "source_type": row["source_type"],
            "output_url": url_for("serve_output", filename=rel, _external=False),
            "created_at": row["created_at"],
            "model": row["model"],
            "duration_ms": int(row["duration_ms"]),
            "conf": float(row["conf"]),
            "iou": float(row["iou"]),
            "classes": json.loads(row["classes_json"] or "[]"),
            "confs": [float(x) for x in json.loads(row["confs_json"] or "[]")],
        }
    )


@app.get("/outputs/<path:filename>")
def serve_output(filename: str):
    return send_from_directory(OUTPUTS_DIR, filename, as_attachment=False)


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5060"))
    app.run(host="0.0.0.0", port=port, debug=True) 