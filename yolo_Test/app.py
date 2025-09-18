import os
import io
import time
import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, List, Tuple

from flask import Flask, request, jsonify, render_template, send_from_directory, url_for, Response
import time as _time

from detector import YOLODetector
import cv2
from face_emotion import FaceEmotionClassifier


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

# Use latest Ultralytics YOLO11 by default for better precision; override via YOLO_MODEL env
_detector_model_default = "yolo11x.pt"
detector = YOLODetector(model_name=os.environ.get("YOLO_MODEL", _detector_model_default))

# Face/emotion classifier (lazy)
_emotion_instance = None

def get_emotion() -> FaceEmotionClassifier:
    global _emotion_instance
    if _emotion_instance is None:
        _emotion_instance = FaceEmotionClassifier(BASE_DIR)
    return _emotion_instance


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


def _open_camera_prefer_avfoundation(preferred_index: int = -1) -> cv2.VideoCapture:
    """Try multiple backends and device indices for macOS reliability."""
    indices = [preferred_index] if preferred_index >= 0 else []
    for idx in [0, 1, 2, 3]:
        if idx not in indices:
            indices.append(idx)
    backends = []
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(cv2.CAP_AVFOUNDATION)
    if hasattr(cv2, "CAP_QT"):
        backends.append(cv2.CAP_QT)
    backends.append(cv2.CAP_ANY)

    for idx in indices:
        for be in backends:
            try:
                cap = cv2.VideoCapture(idx, be)
                if cap is not None and cap.isOpened():
                    return cap
                if cap is not None:
                    cap.release()
            except Exception:
                continue
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            return cap
        if cap is not None:
            cap.release()
    return cv2.VideoCapture(0)


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


@app.post("/api/emotion_detect")
def api_emotion_detect():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    ok, _ = _allowed_file(file.filename)
    if not ok:
        return jsonify({"error": "Unsupported file type"}), 400

    engine = (request.form.get("engine") or "auto").lower()

    # Save upload
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
    safe_name = f"{timestamp}_{os.path.basename(file.filename)}"
    src_path = os.path.join(UPLOADS_DIR, safe_name)
    file.save(src_path)

    # Analyze emotion (images only recommended)
    t0 = time.time()
    try:
        result = get_emotion().analyze_image(src_path, OUTPUTS_DIR, engine=engine)
    except Exception as e:
        return jsonify({"error": f"Emotion analysis failed: {e}"}), 500
    duration_ms = int((time.time() - t0) * 1000)

    rel = os.path.relpath(result["output_path"], OUTPUTS_DIR)
    classes = [f["emotion"] for f in result.get("faces", [])]
    confs = [float(f.get("confidence", 0.0)) for f in result.get("faces", [])]

    history_id = _insert_history(
        source_filename=safe_name,
        source_type=f"emotion_image:{engine}",
        output_relpath=rel.replace("\\", "/"),
        classes=classes,
        confs=confs,
        model=f"emotion:{engine}",
        duration_ms=duration_ms,
        conf=0.0,
        iou=0.0,
    )

    return jsonify(
        {
            "id": history_id,
            "output_url": url_for("serve_output", filename=rel, _external=False),
            "faces": result.get("faces", []),
            "duration_ms": duration_ms,
            "model": f"emotion:{engine}",
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


@app.get("/webcam")
def webcam_stream():
    try:
        conf = float(request.args.get("conf", 0.35))
        iou = float(request.args.get("iou", 0.45))
    except Exception:
        conf = 0.35
        iou = 0.45
    try:
        device = int(request.args.get("device", -1))
    except Exception:
        device = -1

    def gen():
        cap = _open_camera_prefer_avfoundation(device)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 20)
        except Exception:
            pass
        if not cap or not cap.isOpened():
            img = 255 * (1 - (cv2.putText(
                img=cv2.cvtColor((255 * (cv2.getStructuringElement(cv2.MORPH_RECT, (480, 320)))).astype('uint8'), cv2.COLOR_GRAY2BGR),
                text='Camera not available', org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2
            ) or 0))
            ret, buf = cv2.imencode('.jpg', img)
            frame = buf.tobytes() if ret else b''
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Cache-Control: no-store\r\n\r\n" + frame + b"\r\n")
            return
        try:
            detector._ensure_loaded()
            last = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = _time.time()
                if now - last < 0.045:
                    continue
                last = now
                try:
                    results = detector._model.predict(frame, conf=conf, iou=iou, imgsz=640, verbose=False)
                    plotted = results[0].plot() if results and len(results) else frame
                except Exception:
                    plotted = frame
                ret, jpeg = cv2.imencode('.jpg', plotted)
                if not ret:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Cache-Control: no-store\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        finally:
            cap.release()

    resp = Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp


@app.get("/webcam_emotion")
def webcam_emotion_stream():
    engine = (request.args.get("engine") or "auto").lower()
    try:
        device = int(request.args.get("device", -1))
    except Exception:
        device = -1
    def gen():
        cap = _open_camera_prefer_avfoundation(device)
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 20)
        except Exception:
            pass
        if not cap or not cap.isOpened():
            img = 255 * (1 - (cv2.putText(
                img=cv2.cvtColor((255 * (cv2.getStructuringElement(cv2.MORPH_RECT, (480, 320)))).astype('uint8'), cv2.COLOR_GRAY2BGR),
                text='Camera not available', org=(10, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1, color=(0, 0, 255), thickness=2
            ) or 0))
            ret, buf = cv2.imencode('.jpg', img)
            frame = buf.tobytes() if ret else b''
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Cache-Control: no-store\r\n\r\n" + frame + b"\r\n")
            return
        try:
            last = 0.0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now = _time.time()
                if now - last < 0.045:
                    continue
                last = now
                try:
                    plotted, _ = get_emotion().analyze_frame(frame, engine=engine)
                except Exception:
                    plotted = frame
                ret, jpeg = cv2.imencode('.jpg', plotted)
                if not ret:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Cache-Control: no-store\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        finally:
            cap.release()

    resp = Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    return resp

# Diagnostics: test camera indices/backends
@app.get("/debug/cam")
def debug_cam():
    results = []
    backends = []
    if hasattr(cv2, "CAP_AVFOUNDATION"):
        backends.append(("AVFOUNDATION", cv2.CAP_AVFOUNDATION))
    if hasattr(cv2, "CAP_QT"):
        backends.append(("QT", cv2.CAP_QT))
    backends.append(("ANY", cv2.CAP_ANY))
    for idx in [0, 1, 2, 3]:
        row = {"index": idx}
        for name, be in backends:
            try:
                cap = cv2.VideoCapture(idx, be)
                ok = bool(cap and cap.isOpened())
                row[name] = ok
                if cap:
                    cap.release()
            except Exception:
                row[name] = False
        results.append(row)
    return jsonify({"devices": results})


@app.get("/snapshot")
def snapshot():
    try:
        device = int(request.args.get("device", 0))
    except Exception:
        device = 0
    cap = _open_camera_prefer_avfoundation(device)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    except Exception:
        pass
    ok, frame = cap.read() if cap and cap.isOpened() else (False, None)
    tries = 0
    while not ok and tries < 5 and cap:
        tries += 1
        ok, frame = cap.read()
    if cap:
        cap.release()
    if not ok or frame is None:
        return jsonify({"error": f"Failed to read frame from device {device}"}), 500
    ret, jpeg = cv2.imencode('.jpg', frame)
    if not ret:
        return jsonify({"error": "Encode failed"}), 500
    resp = Response(jpeg.tobytes(), mimetype='image/jpeg')
    resp.headers['Cache-Control'] = 'no-store'
    return resp


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5060"))
    app.run(host="0.0.0.0", port=port, debug=True) 