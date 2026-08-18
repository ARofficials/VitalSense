# config.py
import os
import numpy as np

# --- SILENCE LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['GLOG_minloglevel'] = '2'

# --- HARDWARE SETTINGS ---
os.environ["PYTHONMALLOC"] = "malloc"

# --- CONSTANTS ---
CAPTURE_FRAMES = 900  # 30 seconds
BATCH_SIZE = 300
FS_TARGET = 30.0

# Buffer Structure
DTYPE_BUFFER = np.dtype([
    ('fh', np.uint8, (72, 72, 3)), 
    ('time', np.float64)
])

# --- 📁 MODEL PATHS (UPDATED) ---
# We use os.path.join to ensure it works on Windows/Linux/Mac
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# PyTorch Models
TSCAN_PATH = os.path.join(MODELS_DIR, "tscan_Official_best_EPO60.pth")
RESNET_PATH = os.path.join(MODELS_DIR, "hr_resnet_denoise.pth")

# ML Models
BREATHING_MODEL_PATH = os.path.join(MODELS_DIR, "breathing_model.json") # XGBoost
STRESS_MODEL_PATH = os.path.join(MODELS_DIR, "stress_model-2.txt")      # LightGBM