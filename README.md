# Raspberry Pi 5 + NoIR Camera Drowsiness Detection (MRL Dataset)

Real-time driver drowsiness detection system trained on the [Kaggle MRL Eye Dataset (`prasadvpatil/mrl-dataset`)](https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset) and optimized for **Raspberry Pi 5** with an official **Raspberry Pi NoIR Camera Module** (via `Picamera2`).

---

## Step 1: Train the Model on Google Colab (~2–3 mins on T4 GPU)

1. Open [Google Colab](https://colab.research.google.com/) and upload [`train_mrl_colab.ipynb`](./train_mrl_colab.ipynb).
2. Go to **Runtime → Change runtime type → T4 GPU**.
3. Click **Runtime → Run all**:
   - The notebook automatically downloads `prasadvpatil/mrl-dataset` using `kagglehub` (no `kaggle.json` needed).
   - Applies **1-channel Grayscale + CLAHE** preprocessing to match the Raspberry Pi NoIR camera across both daylight and night-time IR LED lighting.
   - Trains `MRLEyeCNN` and exports `mrl_eye_model.onnx` (and `mrl_eye_model.pt`), then automatically downloads `mrl_eye_model.onnx` to your computer.
4. Put the downloaded `mrl_eye_model.onnx` in the same folder as [`app.py`](./app.py).

---

## Step 2: Run on Raspberry Pi 5 (with CSI NoIR Camera)

On **Raspberry Pi OS (Bookworm 64-bit)**, `Picamera2` and `OpenCV` (including `cv2.dnn` for ONNX inference) are available straight from `apt`—meaning you **do not** even need to install heavy PyTorch wheels on the Pi:

```bash
sudo apt update
sudo apt install -y python3-picamera2 python3-opencv python3-numpy
```

Verify your NoIR CSI camera is detected by the Pi 5:
```bash
rpicam-hello --timeout 2000
```

Copy `app.py`, `mrl_eye_model.onnx`, and `face_detection_yunet_2023mar.onnx` to your Raspberry Pi 5 and run:
```bash
python3 app.py --model mrl_eye_model.onnx
```

### Useful CLI Flags
```bash
# Start with IR Night-Vision (CLAHE high-contrast monochrome) view enabled:
python3 app.py --model mrl_eye_model.onnx --night-vision

# Adjust eye-closed duration before triggering DROWSINESS ALERT (default: 1.5 seconds):
python3 app.py --model mrl_eye_model.onnx --drowsy-time 1.2 --threshold 0.55
```

### Live Window Keyboard Shortcuts
- **`q` / `ESC`**: Quit
- **`n`**: Toggle **NoIR Night-Vision Mode** (CLAHE-enhanced monochrome vs raw camera view)
- **`+` / `-`**: Increase / decrease closed-eye sensitivity threshold (`0.20` – `0.90`)
- **`r`**: Reset drowsiness timer and alert banner

