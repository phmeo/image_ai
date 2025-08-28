# AI Image Classifier (Flask + TensorFlow)

Ứng dụng web đơn giản để tải ảnh và phân loại bằng MobileNetV2 (ImageNet). Lưu kết quả vào SQLite để thống kê.

## Cài đặt

```bash
python3 -m venv .venv
source .venv/bin/activate
# Apple Silicon (M1/M2/M3):
#   pip install --upgrade pip
#   pip install tensorflow-macos tensorflow-metal
# Intel CPU only:
#   pip install tensorflow
# Common deps:
#   pip install Flask Pillow numpy
pip install -r requirements.txt
```

## Chạy ứng dụng

```bash
python app.py
```

Mở trình duyệt tới `http://localhost:5000`.

## Cấu trúc

- `app.py`: Flask server và route
- `model.py`: Tải và suy luận mô hình MobileNetV2
- `database.py`: SQLite helpers (khởi tạo, lưu, truy vấn)
- `templates/`: Giao diện HTML (Jinja2)
- `static/`: CSS và JS
- `uploads/`: Ảnh đã tải lên (tự tạo nếu chưa có)

## Lưu ý

- Mô hình dùng trọng số ImageNet, phù hợp demo chung (chó/mèo, đồ vật, v.v.)
- Nếu muốn phân loại domain riêng (hoa, rác tái chế), bạn có thể fine-tune và thay thế `classify_image`.