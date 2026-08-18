# inspect_model.py
import xgboost as xgb
import os

model_path = "models/breathing_model.json"

if os.path.exists(model_path):
    try:
        bst = xgb.Booster()
        bst.load_model(model_path)
        print("\n✅ MODEL LOADED SUCCESSFULLY")
        print("--------------------------------")
        print(f"Features Expected: {len(bst.feature_names)}")
        print("Feature Names (In Order):")
        print(bst.feature_names)
        print("--------------------------------\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
else:
    print(f"❌ File not found: {model_path}")