"""
Raspberry Pi 5 Real-Time Drowsiness Detection (NoIR Camera Optimized)
Trained on the Kaggle MRL Eye Dataset (https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset)

Features:
- Native Raspberry Pi 5 CSI camera support via Picamera2 (with automatic OpenCV webcam fallback for PC testing)
- NoIR (Near-Infrared) camera pipeline: Grayscale + CLAHE contrast normalization so daylight (pinkish cast)
  and night-time IR LED illumination both match the MRL NIR eye training distribution
- Zero-heavy-dependency inference via OpenCV DNN (mrl_eye_model.onnx), with optional ONNXRuntime / PyTorch (.pt) support
- Anthropometric Eye-ROI Locking: keeps eye bounding boxes locked onto closed eyelids even when Haar cascade drops out
- Live HUD with Eye State probabilities, Drowsiness progress bar, FPS counter, and visual DROWSINESS ALERT banner

Controls in Live Window:
  [q] or [ESC] : Quit
  [n]          : Toggle NoIR Night-Vision (CLAHE Enhanced IR view vs Raw Camera view)
  [+] / [-]    : Increase / Decrease eye-closed probability threshold (default 0.55)
  [r]          : Reset drowsiness counter / alert
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Model Definition (Used only if loading a .pt PyTorch checkpoint directly)
# ---------------------------------------------------------------------------
def build_pytorch_model():
    import torch
    import torch.nn as nn

    class MRLEyeCNN(nn.Module):
        """Lightweight CNN tailored for 64x64 1-channel NIR/Grayscale eye crops."""

        def __init__(self, num_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                # Block 1: 64x64 -> 32x32
                nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.15),
                # Block 2: 32x32 -> 16x16
                nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.20),
                # Block 3: 16x16 -> 8x8
                nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25),
                # Block 4: 8x8 -> 1x1
                nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.4),
                nn.Linear(64, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.classifier(self.features(x))

    return MRLEyeCNN()


# ---------------------------------------------------------------------------
# Unified MRL Eye Classifier (ONNX via OpenCV DNN / ONNXRuntime / PyTorch)
# ---------------------------------------------------------------------------
class MRLEyeClassifier:
    """
    Runs inference on a 64x64 grayscale CLAHE-normalized eye crop.
    Class mapping (matches MRL Dataset alphabetical folder order):
      0 -> Close-Eyes (Closed)
      1 -> Open-Eyes  (Open)
    Returns P(Closed) in [0.0, 1.0].
    """

    IMG_SIZE = 64

    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.backend = "none"
        self.net = None
        self.ort_session = None
        self.torch_model = None
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

        self._load_model()

    def _load_model(self) -> None:
        if not self.model_path.exists():
            # Check fallback extension (.onnx vs .pt)
            alt_onnx = self.model_path.with_suffix(".onnx")
            alt_pt = self.model_path.with_suffix(".pt")
            if alt_onnx.exists():
                self.model_path = alt_onnx
            elif alt_pt.exists():
                self.model_path = alt_pt
            else:
                print(
                    f"[WARNING] Model file '{self.model_path}' not found yet.\n"
                    f"          Train the model in Google Colab (train_mrl_colab.ipynb) and place\n"
                    f"          'mrl_eye_model.onnx' in: {Path.cwd()}\n"
                    f"          Running with fallback intensity/eyelid heuristic so you can test your camera."
                )
                self.backend = "heuristic_fallback"
                return

        suffix = self.model_path.suffix.lower()

        if suffix == ".onnx":
            # Prefer OpenCV's built-in DNN module (zero extra install on Raspberry Pi OS)
            try:
                self.net = cv2.dnn.readNetFromONNX(str(self.model_path))
                self.backend = "opencv_dnn_onnx"
                print(f"[INFO] Loaded MRL ONNX model via OpenCV DNN: {self.model_path}")
                return
            except Exception as e:
                print(f"[WARN] OpenCV DNN failed to load ONNX ({e}), trying onnxruntime...")

            try:
                import onnxruntime as ort

                self.ort_session = ort.InferenceSession(
                    str(self.model_path), providers=["CPUExecutionProvider"]
                )
                self.backend = "onnxruntime"
                print(f"[INFO] Loaded MRL ONNX model via ONNXRuntime: {self.model_path}")
                return
            except Exception as e:
                raise RuntimeError(f"Could not load ONNX model {self.model_path}: {e}") from e

        elif suffix in (".pt", ".pth"):
            import torch

            self.torch_model = build_pytorch_model()
            ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=True)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.torch_model.load_state_dict(state_dict)
            self.torch_model.eval()
            self.backend = "pytorch"
            print(f"[INFO] Loaded MRL PyTorch model: {self.model_path}")
        else:
            raise ValueError(f"Unsupported model file extension: {suffix} (use .onnx or .pt)")

    def preprocess(self, eye_gray: np.ndarray) -> np.ndarray:
        """
        Matches training preprocessing in train_mrl_colab.ipynb:
        1. CLAHE equalization (normalizes NoIR lighting / IR LED hot-spots)
        2. Resize to 64x64
        3. Scale to [0, 1] and normalize with mean=0.5, std=0.5 -> [-1, 1]
        4. Shape (1, 1, 64, 64) float32
        """
        if eye_gray.ndim == 3:
            eye_gray = cv2.cvtColor(eye_gray, cv2.COLOR_BGR2GRAY)
        eq = self.clahe.apply(eye_gray)
        resized = cv2.resize(eq, (self.IMG_SIZE, self.IMG_SIZE), interpolation=cv2.INTER_AREA)
        norm = (resized.astype(np.float32) / 255.0 - 0.5) / 0.5
        return norm[np.newaxis, np.newaxis, :, :]  # (1, 1, 64, 64)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - np.max(logits)
        exp_scores = np.exp(shifted)
        return exp_scores / np.sum(exp_scores)

    def predict_closed_prob(self, eye_gray: np.ndarray) -> float:
        """Returns probability [0.0, 1.0] that the eye is CLOSED."""
        if eye_gray is None or eye_gray.size == 0:
            return 0.0

        if self.backend == "heuristic_fallback":
            # Simple vertical gradient energy fallback when testing camera before model is placed
            resized = cv2.resize(eye_gray, (64, 64))
            central = resized[16:48, 12:52]
            sobel_x = np.abs(cv2.Sobel(central, cv2.CV_32F, 1, 0, ksize=3)).mean()
            sobel_y = np.abs(cv2.Sobel(central, cv2.CV_32F, 0, 1, ksize=3)).mean()
            ratio = sobel_x / (sobel_y + 1e-5)
            # Open eyes have strong horizontal iris contrast (higher sobel_x/sobel_y ratio)
            prob_closed = float(np.clip(1.0 - (ratio - 0.35) / 0.55, 0.05, 0.95))
            return prob_closed

        blob = self.preprocess(eye_gray)

        if self.backend == "opencv_dnn_onnx":
            self.net.setInput(blob)
            logits = self.net.forward().flatten()
            probs = self._softmax(logits)
            return float(probs[0])  # Index 0 = Close-Eyes

        if self.backend == "onnxruntime":
            input_name = self.ort_session.get_inputs()[0].name
            logits = self.ort_session.run(None, {input_name: blob})[0].flatten()
            probs = self._softmax(logits)
            return float(probs[0])

        if self.backend == "pytorch":
            import torch

            with torch.no_grad():
                tensor = torch.from_numpy(blob)
                logits = self.torch_model(tensor).numpy().flatten()
                probs = self._softmax(logits)
                return float(probs[0])

        return 0.0


# ---------------------------------------------------------------------------
# Camera Wrapper: Picamera2 (Raspberry Pi 5 CSI NoIR) + OpenCV Fallback
# ---------------------------------------------------------------------------
class CameraStream:
    """
    Uses Picamera2 on Raspberry Pi 5 for CSI NoIR Camera Modules,
    and automatically falls back to OpenCV VideoCapture for USB cameras or PC testing.
    """

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30, camera_index: int = 0):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera_index = camera_index
        self.picam2 = None
        self.cap = None
        self.source_name = "Unknown"

        self._init_camera()

    def _init_camera(self) -> None:
        # 1. Try Picamera2 (native for Raspberry Pi 5 CSI ribbon cameras including NoIR)
        try:
            from picamera2 import Picamera2

            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (self.width, self.height), "format": "RGB888"},
                controls={"FrameDurationLimits": (int(1_000_000 / self.fps), int(1_000_000 / self.fps))},
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.5)  # Warm-up sensor
            self.source_name = "Picamera2 (RPi5 CSI NoIR)"
            print(f"[INFO] Started camera via {self.source_name} at {self.width}x{self.height}")
            return
        except ImportError:
            print("[INFO] 'picamera2' not found; falling back to OpenCV VideoCapture.")
        except Exception as e:
            print(f"[WARN] Picamera2 initialization failed ({e}); falling back to OpenCV VideoCapture.")
            self.picam2 = None

        # 2. Fallback to OpenCV VideoCapture (USB webcam or PC testing)
        backend = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(self.camera_index, backend)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera (tried Picamera2 and cv2.VideoCapture({self.camera_index})).\n"
                "On Raspberry Pi 5, check 'rpicam-hello' and ensure 'python3-picamera2' is installed."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.source_name = f"OpenCV Cam #{self.camera_index}"
        print(f"[INFO] Started camera via {self.source_name}")

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
                # Note: Picamera2 "RGB888" format on little-endian DRM outputs BGR byte order directly for OpenCV
                return True, frame
            except Exception as e:
                print(f"[ERROR] Picamera2 frame capture error: {e}")
                return False, None

        if self.cap is not None:
            return self.cap.read()

        return False, None

    def release(self) -> None:
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------------------
# Face & Eye Tracker with Anthropometric ROI Locking for Closed Eyes
# ---------------------------------------------------------------------------
@dataclass
class EyeDetection:
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) in full-frame coordinates
    prob_closed: float
    is_locked_fallback: bool = False


class FaceAndEyeTracker:
    """
    Detects the driver's face and extracts Left and Right eye patches.
    Primary backend: OpenCV YuNet (`cv2.FaceDetectorYN` with `face_detection_yunet_2023mar.onnx`),
    which directly predicts the face bounding box AND the exact left/right eye center landmarks
    even when the eyelids are completely closed or the head is tilted.
    Fallback backend: OpenCV Haar Cascades (for older OpenCV 4.x builds if YuNet model is absent).
    """

    YUNET_URL = (
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    )
    YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

    def __init__(self, yunet_path: str = YUNET_FILENAME):
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.yunet = None
        self.yunet_input_size = (320, 240)
        self.face_cascade = None
        self.eye_cascade = None
        self.eye_cascade_std = None

        # Smoothed state for jitter-free bounding boxes
        self.smooth_face: Optional[np.ndarray] = None
        self.smooth_left_box: Optional[np.ndarray] = None
        self.smooth_right_box: Optional[np.ndarray] = None
        self.missed_face_frames: int = 0
        self.max_missed_face_frames: int = 8

        self.left_eye_rel = np.array([0.16, 0.24, 0.28, 0.22], dtype=np.float32)
        self.right_eye_rel = np.array([0.56, 0.24, 0.28, 0.22], dtype=np.float32)

        self._init_detector(Path(yunet_path))

    def _init_detector(self, yunet_path: Path) -> None:
        if not yunet_path.exists():
            try:
                import urllib.request

                print(f"[INFO] Downloading YuNet face detector ({self.YUNET_FILENAME})...")
                urllib.request.urlretrieve(self.YUNET_URL, str(yunet_path))
            except Exception as e:
                print(f"[WARN] Could not auto-download YuNet model ({e}).")

        if yunet_path.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(
                    model=str(yunet_path),
                    config="",
                    input_size=self.yunet_input_size,
                    score_threshold=0.60,
                    nms_threshold=0.30,
                    top_k=5,
                )
                print(f"[INFO] Initialized YuNet Face & Eye Landmark Detector: {yunet_path}")
                return
            except Exception as e:
                print(f"[WARN] FaceDetectorYN init failed ({e}), trying Haar Cascade fallback...")

        if hasattr(cv2, "CascadeClassifier") and hasattr(cv2, "data"):
            haar_dir = cv2.data.haarcascades
            self.face_cascade = cv2.CascadeClassifier(
                os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
            )
            self.eye_cascade = cv2.CascadeClassifier(
                os.path.join(haar_dir, "haarcascade_eye_tree_eyeglasses.xml")
            )
            self.eye_cascade_std = cv2.CascadeClassifier(
                os.path.join(haar_dir, "haarcascade_eye.xml")
            )
            print("[INFO] Initialized Haar Cascade Face & Eye Detector.")
        else:
            raise RuntimeError(
                "Neither YuNet (face_detection_yunet_2023mar.onnx) nor cv2.CascadeClassifier is available."
            )

    @staticmethod
    def _box_around_center(
        cx: float, cy: float, side: float, w_img: int, h_img: int
    ) -> np.ndarray:
        half = side * 0.5
        x = max(0.0, min(cx - half, w_img - side))
        y = max(0.0, min(cy - half, h_img - side))
        w = max(20.0, min(side, w_img - x))
        h = max(20.0, min(side, h_img - y))
        return np.array([x, y, w, h], dtype=np.float32)

    def _detect_with_yunet(
        self, eq_gray: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[Tuple[int, int, int, int], bool]], Optional[Tuple[Tuple[int, int, int, int], bool]]]:
        h_img, w_img = eq_gray.shape[:2]
        in_w, in_h = self.yunet_input_size
        small_gray = cv2.resize(eq_gray, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        small_bgr = cv2.cvtColor(small_gray, cv2.COLOR_GRAY2BGR)

        self.yunet.setInputSize((in_w, in_h))
        _, faces = self.yunet.detect(small_bgr)

        if faces is not None and len(faces) > 0:
            # Pick largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            best = faces[0]
            sx = w_img / float(in_w)
            sy = h_img / float(in_h)

            raw_face = np.array(
                [best[0] * sx, best[1] * sy, best[2] * sx, best[3] * sy], dtype=np.float32
            )
            # YuNet landmarks: best[4:6] is right eye (left side of image), best[6:8] is left eye (right side of image)
            eye1_cx, eye1_cy = best[4] * sx, best[5] * sy
            eye2_cx, eye2_cy = best[6] * sx, best[7] * sy

            # Ensure left_eye is always the one with smaller X in image coordinates
            if eye1_cx <= eye2_cx:
                l_cx, l_cy, r_cx, r_cy = eye1_cx, eye1_cy, eye2_cx, eye2_cy
            else:
                l_cx, l_cy, r_cx, r_cy = eye2_cx, eye2_cy, eye1_cx, eye1_cy

            inter_ocular = float(np.hypot(r_cx - l_cx, r_cy - l_cy))
            eye_side = max(28.0, inter_ocular * 0.72, raw_face[2] * 0.28)

            raw_left_box = self._box_around_center(l_cx, l_cy, eye_side, w_img, h_img)
            raw_right_box = self._box_around_center(r_cx, r_cy, eye_side, w_img, h_img)

            if self.smooth_face is None:
                self.smooth_face = raw_face
                self.smooth_left_box = raw_left_box
                self.smooth_right_box = raw_right_box
            else:
                self.smooth_face = 0.65 * raw_face + 0.35 * self.smooth_face
                self.smooth_left_box = 0.65 * raw_left_box + 0.35 * self.smooth_left_box
                self.smooth_right_box = 0.65 * raw_right_box + 0.35 * self.smooth_right_box

            self.missed_face_frames = 0
        else:
            self.missed_face_frames += 1
            if self.missed_face_frames > self.max_missed_face_frames:
                self.smooth_face = None
                self.smooth_left_box = None
                self.smooth_right_box = None
                return None, None, None

        if self.smooth_face is None or self.smooth_left_box is None or self.smooth_right_box is None:
            return None, None, None

        face_tuple = tuple(int(round(v)) for v in self.smooth_face)
        left_tuple = tuple(int(round(v)) for v in self.smooth_left_box)
        right_tuple = tuple(int(round(v)) for v in self.smooth_right_box)
        is_fallback = self.missed_face_frames > 0
        return face_tuple, (left_tuple, is_fallback), (right_tuple, is_fallback)

    def _detect_eye_in_zone(
        self, zone_gray: np.ndarray
    ) -> Optional[Tuple[int, int, int, int]]:
        min_side = max(18, int(min(zone_gray.shape[:2]) * 0.30))
        eyes = self.eye_cascade.detectMultiScale(
            zone_gray, scaleFactor=1.1, minNeighbors=4, minSize=(min_side, min_side)
        )
        if len(eyes) == 0:
            eyes = self.eye_cascade_std.detectMultiScale(
                zone_gray, scaleFactor=1.1, minNeighbors=3, minSize=(min_side, min_side)
            )
        if len(eyes) == 0:
            return None
        eyes = sorted(eyes, key=lambda b: b[2] * b[3], reverse=True)
        return tuple(int(v) for v in eyes[0])

    def locate_face_and_eyes(
        self, gray_frame: np.ndarray
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple[Tuple[int, int, int, int], bool]], Optional[Tuple[Tuple[int, int, int, int], bool]]]:
        """
        Returns:
          face_bbox: (fx, fy, fw, fh) or None
          left_eye_info: ((x, y, w, h), is_locked_fallback) or None
          right_eye_info: ((x, y, w, h), is_locked_fallback) or None
        """
        eq_gray = self.clahe.apply(gray_frame)

        if self.yunet is not None:
            return self._detect_with_yunet(eq_gray)

        h_img, w_img = eq_gray.shape[:2]

        # Downscale by 0.5x for fast Haar face detection
        small = cv2.resize(eq_gray, (w_img // 2, h_img // 2), interpolation=cv2.INTER_LINEAR)
        faces = self.face_cascade.detectMultiScale(
            small, scaleFactor=1.15, minNeighbors=5, minSize=(50, 50)
        )

        if len(faces) > 0:
            faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
            raw_face = np.array(faces[0], dtype=np.float32) * 2.0
            if self.smooth_face is None:
                self.smooth_face = raw_face
            else:
                self.smooth_face = 0.65 * raw_face + 0.35 * self.smooth_face
            self.missed_face_frames = 0
        else:
            self.missed_face_frames += 1
            if self.missed_face_frames > self.max_missed_face_frames:
                self.smooth_face = None
                return None, None, None

        if self.smooth_face is None:
            return None, None, None

        fx, fy, fw, fh = [int(round(v)) for v in self.smooth_face]
        fx = max(0, min(fx, w_img - 1))
        fy = max(0, min(fy, h_img - 1))
        fw = max(20, min(fw, w_img - fx))
        fh = max(20, min(fh, h_img - fy))

        upper_y1 = fy + int(0.16 * fh)
        upper_y2 = fy + int(0.55 * fh)
        mid_x = fx + fw // 2

        left_zone = eq_gray[upper_y1:upper_y2, fx + int(0.08 * fw) : mid_x]
        right_zone = eq_gray[upper_y1:upper_y2, mid_x : fx + int(0.92 * fw)]

        left_det = self._detect_eye_in_zone(left_zone) if left_zone.size > 0 else None
        if left_det is not None:
            lx, ly, lw, lh = left_det
            abs_lx = fx + int(0.08 * fw) + lx
            abs_ly = upper_y1 + ly
            pad = int(0.12 * max(lw, lh))
            abs_lx = max(0, abs_lx - pad)
            abs_ly = max(0, abs_ly - pad)
            lw = min(w_img - abs_lx, lw + 2 * pad)
            lh = min(h_img - abs_ly, lh + 2 * pad)

            cur_rel = np.array(
                [(abs_lx - fx) / fw, (abs_ly - fy) / fh, lw / fw, lh / fh], dtype=np.float32
            )
            self.left_eye_rel = 0.5 * cur_rel + 0.5 * self.left_eye_rel
            left_info = ((abs_lx, abs_ly, lw, lh), False)
        else:
            rx, ry, rw, rh = self.left_eye_rel
            abs_lx = max(0, int(fx + rx * fw))
            abs_ly = max(0, int(fy + ry * fh))
            lw = min(w_img - abs_lx, max(24, int(rw * fw)))
            lh = min(h_img - abs_ly, max(24, int(rh * fh)))
            left_info = ((abs_lx, abs_ly, lw, lh), True)

        right_det = self._detect_eye_in_zone(right_zone) if right_zone.size > 0 else None
        if right_det is not None:
            rx_d, ry_d, rw_d, rh_d = right_det
            abs_rx = mid_x + rx_d
            abs_ry = upper_y1 + ry_d
            pad = int(0.12 * max(rw_d, rh_d))
            abs_rx = max(0, abs_rx - pad)
            abs_ry = max(0, abs_ry - pad)
            rw_d = min(w_img - abs_rx, rw_d + 2 * pad)
            rh_d = min(h_img - abs_ry, rh_d + 2 * pad)

            cur_rel = np.array(
                [(abs_rx - fx) / fw, (abs_ry - fy) / fh, rw_d / fw, rh_d / fh], dtype=np.float32
            )
            self.right_eye_rel = 0.5 * cur_rel + 0.5 * self.right_eye_rel
            right_info = ((abs_rx, abs_ry, rw_d, rh_d), False)
        else:
            rx, ry, rw, rh = self.right_eye_rel
            abs_rx = max(0, int(fx + rx * fw))
            abs_ry = max(0, int(fy + ry * fh))
            rw_d = min(w_img - abs_rx, max(24, int(rw * fw)))
            rh_d = min(h_img - abs_ry, max(24, int(rh * fh)))
            right_info = ((abs_rx, abs_ry, rw_d, rh_d), True)

        return (fx, fy, fw, fh), left_info, right_info


# ---------------------------------------------------------------------------
# HUD Renderer
# ---------------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    face_bbox: Optional[Tuple[int, int, int, int]],
    left_eye: Optional[EyeDetection],
    right_eye: Optional[EyeDetection],
    closed_threshold: float,
    closed_duration: float,
    alert_seconds: float,
    is_drowsy_alert: bool,
    fps: float,
    cam_source: str,
    model_backend: str,
    night_vision_mode: bool,
) -> np.ndarray:
    h, w = frame.shape[:2]
    canvas = frame.copy()

    # Colors (BGR)
    GREEN = (80, 230, 80)
    RED = (50, 50, 255)
    AMBER = (0, 190, 255)
    CYAN = (230, 210, 60)
    DARK_BG = (25, 25, 25)
    WHITE = (245, 245, 245)

    # 1. Draw Face & Eye Bounding Boxes
    if face_bbox is not None:
        fx, fy, fw, fh = face_bbox
        face_color = RED if is_drowsy_alert else CYAN
        cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), face_color, 2)

        for label, eye in (("L", left_eye), ("R", right_eye)):
            if eye is None:
                continue
            ex, ey, ew, eh = eye.bbox
            is_closed = eye.prob_closed >= closed_threshold
            color = RED if is_closed else GREEN
            cv2.rectangle(canvas, (ex, ey), (ex + ew, ey + eh), color, 2)

            state_str = "CLOSED" if is_closed else "OPEN"
            conf = eye.prob_closed if is_closed else (1.0 - eye.prob_closed)
            tag = f"{label}: {state_str} {int(conf * 100)}%"
            cv2.putText(
                canvas,
                tag,
                (ex, max(18, ey - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
                cv2.LINE_AA,
            )

    # 2. Top Telemetry Panel
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (w, 68), DARK_BG, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)

    status_text = (
        "STATE: DROWSY ALERT!"
        if is_drowsy_alert
        else ("STATE: EYES CLOSING..." if closed_duration > 0.25 else (
            "STATE: AWAKE / ALERT" if face_bbox is not None else "STATE: NO FACE DETECTED"
        ))
    )
    status_color = RED if is_drowsy_alert else (AMBER if closed_duration > 0.25 else GREEN)
    if face_bbox is None:
        status_color = AMBER

    cv2.putText(
        canvas, status_text, (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.70, status_color, 2, cv2.LINE_AA
    )

    info_line = (
        f"FPS: {fps:4.1f} | Cam: {cam_source} | Model: {model_backend} | "
        f"Thr: {closed_threshold:.2f} | IR-Enh: {'ON' if night_vision_mode else 'OFF'}"
    )
    cv2.putText(
        canvas, info_line, (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, WHITE, 1, cv2.LINE_AA
    )

    # 3. Drowsiness Progress Bar (Top Right)
    bar_w, bar_h = 170, 16
    bar_x, bar_y = w - bar_w - 14, 12
    progress = float(np.clip(closed_duration / max(0.1, alert_seconds), 0.0, 1.0))
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (70, 70, 70), -1)
    fill_w = int(bar_w * progress)
    bar_color = RED if progress >= 1.0 else (AMBER if progress >= 0.4 else GREEN)
    if fill_w > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), WHITE, 1)
    cv2.putText(
        canvas,
        f"Closed: {closed_duration:.1f}s / {alert_seconds:.1f}s",
        (bar_x, bar_y + bar_h + 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    # 4. Flashing Full-Width DROWSINESS ALERT Banner
    if is_drowsy_alert:
        flash_on = int(time.time() * 6) % 2 == 0
        banner_bg = RED if flash_on else (0, 0, 160)
        cv2.rectangle(canvas, (0, h - 68), (w, h), banner_bg, -1)
        cv2.rectangle(canvas, (3, 3), (w - 3, h - 3), RED, 4)
        alert_msg = "DROWSINESS DETECTED! WAKE UP!"
        (tw, th), _ = cv2.getTextSize(alert_msg, cv2.FONT_HERSHEY_SIMPLEX, 0.90, 3)
        cv2.putText(
            canvas,
            alert_msg,
            ((w - tw) // 2, h - 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            WHITE,
            3,
            cv2.LINE_AA,
        )
    else:
        # Footer controls hint
        footer = "[Q] Quit   [N] Toggle NoIR Night-Vision   [+/-] Threshold   [R] Reset"
        cv2.putText(
            canvas,
            footer,
            (14, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

    return canvas


# ---------------------------------------------------------------------------
# Main Application Loop
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Raspberry Pi 5 + NoIR Camera Drowsiness Detection (MRL Dataset)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mrl_eye_model.onnx",
        help="Path to trained MRL model (.onnx or .pt). Default: mrl_eye_model.onnx",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Probability threshold above which an eye is considered CLOSED (default: 0.55)",
    )
    parser.add_argument(
        "--drowsy-time",
        type=float,
        default=1.5,
        help="Consecutive seconds eyes must remain closed to trigger DROWSINESS ALERT (default: 1.5s)",
    )
    parser.add_argument("--width", type=int, default=640, help="Camera frame width (default: 640)")
    parser.add_argument("--height", type=int, default=480, help="Camera frame height (default: 480)")
    parser.add_argument("--fps", type=int, default=30, help="Target camera FPS (default: 30)")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="OpenCV camera index if Picamera2 is not used (default: 0)",
    )
    parser.add_argument(
        "--night-vision",
        action="store_true",
        help="Start in CLAHE infrared night-vision display mode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    classifier = MRLEyeClassifier(args.model)
    tracker = FaceAndEyeTracker()
    camera = CameraStream(
        width=args.width, height=args.height, fps=args.fps, camera_index=args.camera_index
    )

    closed_threshold = float(args.threshold)
    alert_seconds = float(args.drowsy_time)
    night_vision_mode = bool(args.night_vision)

    # Temporal smoothing state
    ema_closed_prob = 0.0
    closed_duration = 0.0
    last_tick = time.perf_counter()
    fps_ema = 30.0
    display_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

    window_name = "RPi 5 Drowsiness Detection (MRL NoIR)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print("\n[READY] Drowsiness Detection running. Press 'q' in the video window to exit.")

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("[WARN] Empty camera frame, retrying...")
                time.sleep(0.05)
                continue

            now = time.perf_counter()
            dt = max(1e-4, min(0.5, now - last_tick))
            last_tick = now
            fps_ema = 0.90 * fps_ema + 0.10 * (1.0 / dt)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Optional NoIR Night-Vision display mode (converts pinkish NoIR or dim IR into high-contrast monochrome)
            display_frame = (
                cv2.cvtColor(display_clahe.apply(gray), cv2.COLOR_GRAY2BGR)
                if night_vision_mode
                else frame
            )

            face_bbox, left_info, right_info = tracker.locate_face_and_eyes(gray)

            left_eye_det: Optional[EyeDetection] = None
            right_eye_det: Optional[EyeDetection] = None

            if face_bbox is not None and left_info is not None and right_info is not None:
                (lx, ly, lw, lh), l_locked = left_info
                (rx, ry, rw, rh), r_locked = right_info

                left_crop = gray[ly : ly + lh, lx : lx + lw]
                right_crop = gray[ry : ry + rh, rx : rx + rw]

                p_left_closed = classifier.predict_closed_prob(left_crop)
                p_right_closed = classifier.predict_closed_prob(right_crop)

                left_eye_det = EyeDetection((lx, ly, lw, lh), p_left_closed, l_locked)
                right_eye_det = EyeDetection((rx, ry, rw, rh), p_right_closed, r_locked)

                # Combined bilateral eye closed probability
                mean_closed_prob = 0.5 * (p_left_closed + p_right_closed)
                ema_closed_prob = 0.60 * mean_closed_prob + 0.40 * ema_closed_prob

                if ema_closed_prob >= closed_threshold:
                    closed_duration += dt
                else:
                    # Decay quickly when eyes open so brief blinks do not accumulate
                    closed_duration = max(0.0, closed_duration - 2.5 * dt)
            else:
                closed_duration = max(0.0, closed_duration - 1.5 * dt)

            is_drowsy_alert = closed_duration >= alert_seconds

            hud_frame = draw_hud(
                frame=display_frame,
                face_bbox=face_bbox,
                left_eye=left_eye_det,
                right_eye=right_eye_det,
                closed_threshold=closed_threshold,
                closed_duration=closed_duration,
                alert_seconds=alert_seconds,
                is_drowsy_alert=is_drowsy_alert,
                fps=fps_ema,
                cam_source=camera.source_name,
                model_backend=classifier.backend,
                night_vision_mode=night_vision_mode,
            )

            cv2.imshow(window_name, hud_frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("n"):
                night_vision_mode = not night_vision_mode
            elif key in (ord("+"), ord("=")):
                closed_threshold = min(0.90, round(closed_threshold + 0.05, 2))
            elif key in (ord("-"), ord("_")):
                closed_threshold = max(0.20, round(closed_threshold - 0.05, 2))
            elif key == ord("r"):
                closed_duration = 0.0
                ema_closed_prob = 0.0

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

