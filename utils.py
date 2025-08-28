import os
from typing import Set

ALLOWED_EXTENSIONS: Set[str] = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}


def allowed_file(filename: str) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def ensure_directories(paths):
    for path in paths:
        os.makedirs(path, exist_ok=True) 