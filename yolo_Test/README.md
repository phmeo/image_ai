# YOLO Detection App (yolo_Test)

A self-contained Flask app for AI object detection using Ultralytics YOLOv8. Supports images and videos, stores annotated outputs and a searchable history.

## Features
- Upload image/video for detection
- Adjustable confidence and IoU thresholds
- Annotated outputs saved under `outputs/`
- History stored in SQLite (`history.db`)
- Simple web frontend

## Setup

```bash
cd yolo_Test
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If PyTorch is not installed automatically with `ultralytics`, follow the official instructions for your platform, or run:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
export YOLO_MODEL=yolov8n.pt  # or yolov8s.pt / yolov8m.pt / yolov8l.pt / yolov8x.pt
python app.py
```

The app listens on `http://127.0.0.1:5060`.

## API
- `POST /api/detect` with form-data `file=<image|video>`, optional `conf`, `iou`
- `GET /api/history` list recent detections
- `GET /api/history/<id>` details for an entry
- `GET /outputs/<path>` serves saved annotated files

## Notes
- Outputs are stored in `outputs/<uploaded_name_without_ext>/pred/`.
- Uploaded originals are stored in `uploads/`.
- Set `YOLO_MODEL` to pick a different model size. Smaller models are faster; larger models can be more precise. 