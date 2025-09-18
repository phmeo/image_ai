import os
import io
import time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np

try:
	import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
	ort = None  # lazy error if used without install

import requests

# Optional mediapipe for heuristic multi-emotion
try:
	import mediapipe as mp  # type: ignore
except Exception:  # pragma: no cover
	mp = None

# Optional FER library (7 emotions); heavy deps, so optional
try:
	from fer import FER  # type: ignore
except Exception:  # pragma: no cover
	FER = None


EMOTION_LABELS = [
	"neutral",
	"happiness",
	"surprise",
	"sadness",
	"anger",
	"disgust",
	"fear",
	"contempt",
]


@dataclass
class FaceBox:
	x: int
	y: int
	w: int
	h: int


class FaceEmotionClassifier:
	def __init__(self, base_dir: str) -> None:
		self._models_dir = os.path.join(base_dir, "models")
		os.makedirs(self._models_dir, exist_ok=True)

		# Face detector (Haar cascade - lightweight)
		self._face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
		if not os.path.exists(self._face_cascade_path):
			raise RuntimeError("OpenCV haarcascade files not found. Ensure opencv is correctly installed.")
		self._face_cascade = cv2.CascadeClassifier(self._face_cascade_path)

		# Optional smile detector for fallback emotion (happiness vs neutral)
		self._smile_cascade_path = cv2.data.haarcascades + "haarcascade_smile.xml"
		self._smile_cascade = cv2.CascadeClassifier(self._smile_cascade_path) if os.path.exists(self._smile_cascade_path) else None

		# MediaPipe FaceMesh (heuristic emotions)
		self._mp_facemesh = None
		if mp is not None:
			try:
				self._mp_facemesh = mp.solutions.face_mesh.FaceMesh(
					static_image_mode=False,
					max_num_faces=5,
					refine_landmarks=True,
					min_detection_confidence=0.5,
					min_tracking_confidence=0.5,
				)
			except Exception:
				self._mp_facemesh = None

		# Optional FER engine
		self._fer = None
		if FER is not None:
			try:
				self._fer = FER(mtcnn=False)  # use OpenCV face detection to avoid heavy deps
			except Exception:
				self._fer = None

		# Emotion model (ONNX FER+)
		self._use_onnx = False
		self._emotion_session = None
		self._input_name = None
		self._output_name = None

		env_override = os.environ.get("EMOTION_MODEL_PATH", "").strip()
		model_path = env_override if env_override else os.path.join(self._models_dir, "emotion-ferplus-8.onnx")

		if ort is not None:
			try:
				if not env_override and not os.path.exists(model_path):
					self._emotion_model_path = model_path
					self._download_emotion_model()
				else:
					self._emotion_model_path = model_path
				providers = ["CPUExecutionProvider"]
				self._emotion_session = ort.InferenceSession(self._emotion_model_path, providers=providers)
				self._input_name = self._emotion_session.get_inputs()[0].name
				self._output_name = self._emotion_session.get_outputs()[0].name
				self._use_onnx = True
			except Exception:
				self._use_onnx = False
		else:
			self._use_onnx = False

	def _download_emotion_model(self) -> None:
		huggingface_url = "https://huggingface.co/onnx-community/emotion-ferplus/resolve/main/emotion-ferplus-8.onnx?download=true"
		hf_token = os.environ.get("HF_TOKEN", "").strip()
		headers = {"User-Agent": "yolo-test/1.0"}
		if hf_token:
			headers["Authorization"] = f"Bearer {hf_token}"

		candidate_urls = [
			"https://raw.githubusercontent.com/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
			"https://media.githubusercontent.com/media/onnx/models/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
			"https://cdn.jsdelivr.net/gh/onnx/models@main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx",
			"https://github.com/onnx/models/blob/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx?raw=1",
			huggingface_url,
		]
		last_err: Exception | None = None
		for url in candidate_urls:
			try:
				with requests.get(url, timeout=150, stream=True, headers=headers) as resp:
					resp.raise_for_status()
					tmp_path = self._emotion_model_path + ".part"
					with open(tmp_path, "wb") as f:
						for chunk in resp.iter_content(chunk_size=1024 * 256):
							if chunk:
								f.write(chunk)
					os.replace(tmp_path, self._emotion_model_path)
					if os.path.getsize(self._emotion_model_path) < 100_000:
						raise RuntimeError("Downloaded file too small; likely an HTML page or LFS pointer")
					return
			except Exception as e:
				last_err = e
				continue
		raise RuntimeError(f"Failed to download emotion model. Tried {len(candidate_urls)} URLs. Last error: {last_err}")

	def _detect_faces(self, gray: np.ndarray) -> List[FaceBox]:
		faces = self._face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))
		boxes: List[FaceBox] = []
		for (x, y, w, h) in faces:
			boxes.append(FaceBox(int(x), int(y), int(w), int(h)))
		return boxes

	def _preprocess_emotion(self, face_bgr: np.ndarray) -> np.ndarray:
		gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
		gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
		arr = gray.astype(np.float32) / 255.0
		arr = np.expand_dims(arr, axis=(0, 1))  # [1,1,64,64]
		return arr

	def _infer_emotion_fallback(self, face_bgr: np.ndarray) -> Tuple[str, float, List[float]]:
		if self._smile_cascade is None:
			return "neutral", 0.5, []
		gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
		smiles = self._smile_cascade.detectMultiScale(
			gray,
			scaleFactor=1.7,
			minNeighbors=22,
			minSize=(25, 25),
		)
		if len(smiles) > 0:
			areas = [(w * h) for (_, _, w, h) in smiles]
			rel = min(1.0, max(areas) / (face_bgr.shape[0] * face_bgr.shape[1] / 4.0)) if areas else 0.5
			conf = float(0.6 + 0.4 * rel)
			return "happiness", conf, []
		return "neutral", 0.6, []

	def _infer_emotion(self, face_bgr: np.ndarray) -> Tuple[str, float, List[float]]:
		if self._use_onnx and self._emotion_session is not None:
			input_blob = self._preprocess_emotion(face_bgr)
			outputs = self._emotion_session.run([self._output_name], {self._input_name: input_blob})
			logits = outputs[0][0]
			exps = np.exp(logits - np.max(logits))
			probs = exps / np.sum(exps)
			idx = int(np.argmax(probs))
			label = EMOTION_LABELS[idx]
			conf = float(probs[idx])
			return label, conf, probs.tolist()
		return self._infer_emotion_fallback(face_bgr)

	def _heuristic_emotion_from_landmarks(self, pts: List[Tuple[float, float]]) -> Tuple[str, float]:
		"""Map a few geometric ratios to emotions: happiness, surprise, anger, neutral."""
		def dist(a, b):
			return float(np.hypot(a[0] - b[0], a[1] - b[1]))
		# Key indices from MediaPipe FaceMesh
		try:
			upper_lip = pts[13]; lower_lip = pts[14]
			mouth_left = pts[78]; mouth_right = pts[308]
			left_eye_up = pts[159]; left_eye_down = pts[145]
			right_eye_up = pts[386]; right_eye_down = pts[374]
			brow_left = pts[70]; eye_left_center = pts[468] if len(pts) > 468 else pts[33]
		except Exception:
			return "neutral", 0.5
		mouth_open = dist(upper_lip, lower_lip)
		mouth_width = max(1e-3, dist(mouth_left, mouth_right))
		mo_ratio = mouth_open / mouth_width  # surprise proxy
		left_eye_open = dist(left_eye_up, left_eye_down)
		right_eye_open = dist(right_eye_up, right_eye_down)
		eye_open = (left_eye_open + right_eye_open) / 2.0
		brow_eye = dist(brow_left, eye_left_center)
		# Normalize ratios
		open_norm = mo_ratio
		angry_proxy = 1.0 / max(1e-3, brow_eye)  # smaller distance => larger value
		# Rules
		if open_norm > 0.35 and eye_open > 3.0:  # mouth wide + eyes open
			return "surprise", min(0.95, open_norm)
		if open_norm > 0.22:
			return "happiness", min(0.9, 0.6 + 0.8 * (open_norm - 0.22))
		if angry_proxy > 0.08 and eye_open < 2.0:
			return "anger", min(0.85, angry_proxy)
		return "neutral", 0.6

	def _analyze_with_fer(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
		if self._fer is None:
			raise RuntimeError("FER engine not available")
		res = self._fer.detect_emotions(frame_bgr)
		annotated = frame_bgr.copy()
		items: List[Dict[str, Any]] = []
		for it in res:
			x, y, w, h = it.get('box', [0, 0, 0, 0])
			emo: Dict[str, float] = it.get('emotions', {})
			if not emo:
				continue
			label = max(emo, key=emo.get)
			conf = float(emo[label])
			cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
			cv2.putText(annotated, f"{label} {conf:.2f}", (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
			items.append({
				"box": {"x": x, "y": y, "w": w, "h": h},
				"emotion": label,
				"confidence": conf,
				"probs": [],
			})
		return annotated, items

	def analyze_frame(self, frame_bgr: np.ndarray, engine: Optional[str] = None) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
		engine = (engine or "auto").lower()
		# Engine priority
		if engine == "fer":
			return self._analyze_with_fer(frame_bgr)
		if engine == "onnx":
			# Use ONNX per-face
			return self._analyze_haar_with("onnx", frame_bgr)
		if engine == "mp":
			out = self._analyze_with_mediapipe(frame_bgr)
			if out is not None:
				return out
			return self._analyze_haar_with("fallback", frame_bgr)
		if engine == "smile":
			return self._analyze_haar_with("fallback", frame_bgr)
		# auto: prefer ONNX -> FER -> MediaPipe -> fallback
		if self._use_onnx:
			return self._analyze_haar_with("onnx", frame_bgr)
		if self._fer is not None:
			return self._analyze_with_fer(frame_bgr)
		mp_out = self._analyze_with_mediapipe(frame_bgr)
		if mp_out is not None:
			return mp_out
		return self._analyze_haar_with("fallback", frame_bgr)

	def _analyze_with_mediapipe(self, frame_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, List[Dict[str, Any]]]]:
		if self._mp_facemesh is None:
			return None
		try:
			h, w = frame_bgr.shape[:2]
			results: List[Dict[str, Any]] = []
			annotated = frame_bgr.copy()
			rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
			res = self._mp_facemesh.process(rgb)
			if not res.multi_face_landmarks:
				return None
			for fl in res.multi_face_landmarks:
				pts = [(lm.x * w, lm.y * h) for lm in fl.landmark]
				xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
				x0, y0, x1, y1 = int(max(0, min(xs))), int(max(0, min(ys))), int(min(w, max(xs))), int(min(h, max(ys)))
				label, conf = self._heuristic_emotion_from_landmarks(pts)
				cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
				cv2.putText(annotated, f"{label} {conf:.2f}", (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
				results.append({
					"box": {"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0},
					"emotion": label,
					"confidence": float(conf),
					"probs": [],
				})
			return annotated, results
		except Exception:
			return None

	def _analyze_haar_with(self, mode: str, frame_bgr: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
		gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
		faces = self._detect_faces(gray)
		results: List[Dict[str, Any]] = []
		annotated = frame_bgr.copy()
		for fb in faces:
			x0, y0, x1, y1 = fb.x, fb.y, fb.x + fb.w, fb.y + fb.h
			face_roi = frame_bgr[y0:y1, x0:x1]
			if face_roi.size == 0:
				continue
			try:
				if mode == "onnx":
					label, conf, probs = self._infer_emotion(face_roi)
				else:
					label, conf, probs = self._infer_emotion_fallback(face_roi)
			except Exception:
				label, conf, probs = "unknown", 0.0, []
			cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 255, 0), 2)
			cv2.putText(annotated, f"{label} {conf:.2f}", (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
			results.append(
				{
					"box": {"x": fb.x, "y": fb.y, "w": fb.w, "h": fb.h},
					"emotion": label,
					"confidence": conf,
					"probs": probs,
				}
			)
		return annotated, results

	def analyze_image(self, image_path: str, outputs_root: str, engine: Optional[str] = None) -> Dict[str, Any]:
		img = cv2.imread(image_path)
		if img is None:
			raise RuntimeError("Failed to read image for emotion analysis")
		annotated, results = self.analyze_frame(img, engine=engine)

		run_name = os.path.splitext(os.path.basename(image_path))[0] + "_emotion"
		save_project = os.path.join(outputs_root, run_name)
		os.makedirs(save_project, exist_ok=True)
		out_path = os.path.join(save_project, os.path.basename(image_path))
		cv2.imwrite(out_path, annotated)
		return {"output_path": out_path, "faces": results} 