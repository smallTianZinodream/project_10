@echo off
cd /d "%~dp0"
if not exist "models\Xenova\yolov9-c_all\onnx\model_quantized.onnx" (
  echo Downloading YOLO weights via hf-mirror ...
  python scripts\download_models.py
)
if not exist "vendor\ort\ort-wasm-simd-threaded.jsep.wasm" (
  echo Downloading ONNX Runtime WASM ...
  python scripts\download_ort.py
)
echo Open http://127.0.0.1:8765
python -m http.server 8765
