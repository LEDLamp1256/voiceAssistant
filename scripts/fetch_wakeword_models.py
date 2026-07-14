"""
Idempotent fetch of openWakeWord's default model set.
Run via setup_models.ps1. Safe to re-run — skips files that already
exist and are non-zero size.
"""
from pathlib import Path
from openwakeword.utils import download_models

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Pulls: melspectrogram.onnx, embedding_model.onnx, and the requested
# wakeword model(s) — pinned to whatever openwakeword version is installed,
# so you never get a URL/version mismatch.
download_models(model_names=["hey_jarvis"], target_directory=str(MODELS_DIR))

print(f"[OK] Models fetched into {MODELS_DIR}")