import torch
import os

MODEL_PATH = r"d:\CatalogForProj\PythonProj\2026MCM\model\best_physics_model_final.pth"

def check_model():
    if not os.path.exists(MODEL_PATH):
        print(f"File not found: {MODEL_PATH}")
        return

    try:
        state_dict = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
        print("Model loaded successfully.")
        
        # Check a key parameter to infer d_model
        # input_proj.weight shape should be [d_model, feature_dim]
        if 'input_proj.weight' in state_dict:
            weight = state_dict['input_proj.weight']
            print(f"input_proj.weight shape: {weight.shape}")
            if weight.shape[0] == 256:
                print("CONFIRMED: Model is d_model=256 (RTX 5090 Config)")
            else:
                print(f"WARNING: Model d_model seems to be {weight.shape[0]}, expected 256.")
        else:
            print("Could not find input_proj.weight to verify dimensions.")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    check_model()
