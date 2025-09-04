import json
import os
import time
from datetime import datetime

from flask import Flask, flash, redirect, render_template, request, url_for, send_from_directory, jsonify
from PIL import Image
from werkzeug.utils import secure_filename

from database import (
    get_label_counts,
    get_recent_predictions,
    initialize_database,
    insert_prediction,
)
from model import classify_image, list_available_models, get_model_info
from utils import allowed_file, ensure_directories
from detector import detect_objects


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
DATABASE_PATH = os.path.join(BASE_DIR, "db.sqlite3")

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_DIR
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB
app.config["ASSET_VERSION"] = str(int(time.time()))
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.context_processor
def inject_asset_version():
    return {"ASSET_VERSION": app.config.get("ASSET_VERSION", "0")}


ensure_directories([UPLOAD_DIR, os.path.join(BASE_DIR, "templates"), os.path.join(BASE_DIR, "static")])
initialize_database(DATABASE_PATH)


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", available_models=list_available_models(), model_info=get_model_info())


@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


def _save_upload(file_storage):
    original_filename = secure_filename(file_storage.filename)
    timestamp = int(time.time())
    name, ext = os.path.splitext(original_filename)
    stored_filename = f"{name}-{timestamp}{ext.lower()}"
    stored_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_filename)
    file_storage.save(stored_path)
    return stored_filename, stored_path


def _perform_prediction(file_storage, model_name: str, top_k: int, min_prob: float):
    stored_filename, stored_path = _save_upload(file_storage)

    with Image.open(stored_path) as image:
        predictions = classify_image(image, model_name=model_name, top_k=top_k)

    if min_prob > 0:
        predictions = [(l, p) for (l, p) in predictions if p >= min_prob]
        if len(predictions) == 0:
            predictions = [("No result above threshold", 0.0)]

    top1_label, top1_prob = predictions[0]
    insert_prediction(
        DATABASE_PATH,
        filename=stored_filename,
        top1_label=top1_label,
        top1_confidence=float(top1_prob),
        predictions=predictions,
        model_name=model_name,
    )

    return stored_filename, predictions


def _perform_detection(file_storage, min_score: float = 0.4):
    stored_filename, stored_path = _save_upload(file_storage)
    with Image.open(stored_path) as image:
        det = detect_objects(image, score_threshold=min_score)
    return stored_filename, det


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("Vui lòng chọn một ảnh để tải lên.")
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        flash("Tên tệp không hợp lệ.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Định dạng tệp không được hỗ trợ.")
        return redirect(url_for("index"))

    mode = request.form.get("mode") or "classify"

    if mode == "detect":
        try:
            stored_filename, det = _perform_detection(file, min_score=0.35)
        except Exception:
            flash("Không thể phát hiện vật thể. Vui lòng thử ảnh khác.")
            return redirect(url_for("index"))
        return render_template(
            "index.html",
            available_models=list_available_models(),
            model_info=get_model_info(),
            detection={
                "filename": stored_filename,
                "width": det["width"],
                "height": det["height"],
                "detections": [
                    {
                        "box": d.get("box", [0,0,0,0]),
                        "boxn": d.get("boxn", [0,0,0,0]),
                        "score": round(float(d.get("score", 0.0)) * 100.0, 2),
                        "class_id": d.get("class_id", -1),
                        "label": d.get("label", f"id {d.get('class_id', -1)}"),
                    }
                    for d in det["detections"]
                ],
            },
        )

    model_name = request.form.get("model") or "efficientnet_v2_b3"
    try:
        top_k = int(request.form.get("top_k") or 5)
    except Exception:
        top_k = 5
    try:
        min_prob = float(request.form.get("min_prob") or 0)
    except Exception:
        min_prob = 0.0

    try:
        stored_filename, predictions = _perform_prediction(file, model_name, max(1, min(top_k, 5)), max(0.0, min(min_prob, 1.0)))
    except Exception:
        flash("Không thể xử lý ảnh. Vui lòng thử ảnh khác.")
        return redirect(url_for("index"))

    return render_template(
        "index.html",
        available_models=list_available_models(),
        model_info=get_model_info(),
        result={
            "filename": stored_filename,
            "top1_label": predictions[0][0],
            "top1_prob": round(float(predictions[0][1]) * 100.0, 2),
            "predictions": [{"label": l, "prob": round(float(p) * 100.0, 2)} for (l, p) in predictions],
            "model": model_name,
        },
    )


@app.route("/stats", methods=["GET"])
def stats():
    label_counts = get_label_counts(DATABASE_PATH)
    recent = get_recent_predictions(DATABASE_PATH, limit=20)
    for item in recent:
        try:
            item["predictions"] = json.loads(item.get("predictions_json") or "[]")
        except Exception:
            item["predictions"] = []
    return render_template("stats.html", label_counts=label_counts, recent=recent)


# Simple JSON API
@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify({"models": list_available_models(), "info": get_model_info()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "image" not in request.files:
        return jsonify({"error": "missing image"}), 400
    file = request.files["image"]
    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"error": "invalid file"}), 400

    model_name = request.form.get("model") or request.args.get("model") or "efficientnet_v2_b3"
    top_k = int(request.form.get("top_k") or request.args.get("top_k") or 5)
    min_prob = float(request.form.get("min_prob") or request.args.get("min_prob") or 0)

    try:
        stored_filename, predictions = _perform_prediction(file, model_name, max(1, min(top_k, 5)), max(0.0, min(min_prob, 1.0)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "filename": stored_filename,
        "model": model_name,
        "predictions": [{"label": l, "prob": p} for (l, p) in predictions],
        "top1": {"label": predictions[0][0], "prob": predictions[0][1]},
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=True) 