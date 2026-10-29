@echo off
cd /d "%~dp0"
.\.venv\Scripts\python.exe app.py --model mrl_eye_model.onnx
pause

