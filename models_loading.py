# models_loading.py
import os
import torch
import xgboost as xgb
import config
from models_definitions import TSCAN, HeartRateResNet

class ModelLoader:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"⚙️  Compute Device: {self.device}")

    def load_tscan(self):
        model = TSCAN(frames=config.BATCH_SIZE).to(self.device)
        if os.path.exists(config.TSCAN_PATH):
            model.load_state_dict(torch.load(config.TSCAN_PATH, map_location=self.device))
        else:
            print("⚠️  TSCAN weights not found. Using random init.")
        model.eval()
        return model

    def load_resnet(self):
        model = HeartRateResNet().to(self.device)
        if os.path.exists(config.RESNET_PATH):
            model.load_state_dict(torch.load(config.RESNET_PATH, map_location=self.device))
        model.eval()
        return model

    def load_breathing_model(self):
        """Loads XGBoost model for Breathing Rate"""
        if os.path.exists(config.BREATHING_MODEL_PATH):
            try:
                bst = xgb.Booster()
                bst.load_model(config.BREATHING_MODEL_PATH)
                print(f"✅ Breathing Model Loaded (XGBoost): {os.path.basename(config.BREATHING_MODEL_PATH)}")
                return bst
            except Exception as e:
                print(f"❌ Breathing Model Failed: {e}")
        return None

    def load_stress_model(self):
        """Loads LightGBM model for Stress Detection"""
        if os.path.exists(config.STRESS_MODEL_PATH):
            try:
                import lightgbm as lgb
                bst = lgb.Booster(model_file=config.STRESS_MODEL_PATH)
                print(f"✅ Stress Model Loaded (LightGBM): {os.path.basename(config.STRESS_MODEL_PATH)}")
                return bst
            except ImportError:
                print("❌ LightGBM not installed. Run: pip install lightgbm")
            except Exception as e:
                print(f"❌ Stress Model Failed: {e}")
        return None