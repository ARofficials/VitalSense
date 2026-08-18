import os
import sys
import numpy as np

# --- SILENCE LOGS ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ['GLOG_minloglevel'] = '2'
os.environ["PYTHONMALLOC"] = "malloc"

# --- PYINSTALLER RESOURCE PATH FIX ---
def get_resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- CONSTANTS ---
CAPTURE_FRAMES = 900 
BATCH_SIZE = 300
FS_TARGET = 30.0

DTYPE_BUFFER = np.dtype([
    ('fh', np.uint8, (72, 72, 3)), 
    ('time', np.float64)
])

# --- 📁 MODEL PATHS (UPDATED) ---
TSCAN_PATH = get_resource_path(os.path.join("models", "tscan_Official_best_EPO60.pth"))
RESNET_PATH = get_resource_path(os.path.join("models", "hr_resnet_denoise.pth"))
STRESS_MODEL_PATH = get_resource_path(os.path.join("models", "stress_model-2.txt"))

# NEW BREATHING MODEL (PyTorch)
BREATHING_MODEL_PATH = get_resource_path(os.path.join("models", "resp_ppg_model2.pt"))