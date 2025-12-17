import sys
import os
import cv2
import numpy as np

# Add root to path
sys.path.append(os.getcwd())

def test_inference():
    print("Testing inference...")
    
    # Create dummy image (blue square)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (255, 0, 0) # BGR
    
    try:
        from deployment import inference
        class_id, label, conf = inference.predict(img)
        print(f"Success! Prediction: ID={class_id}, Label={label}, Conf={conf}")
        
        valid_types = (
            isinstance(class_id, (int, np.integer)) and 
            isinstance(label, str) and 
            isinstance(conf, float)
        )
        
        if valid_types:
            print("Types check passed.")
        else:
            print(f"Type mismatch: {type(class_id)}, {type(label)}, {type(conf)}")
            
    except OSError as e:
        print(f"Inference failed (Environment Issue): {e}")
        print("Tip: If this is a DLL error, try reinstalling PyTorch via Conda.")
    except Exception as e:
        print(f"Inference failed: {e}")
        if "Model not found" in str(e):
            print("Explanation: Model file missing. Training script likely still running.")

if __name__ == "__main__":
    test_inference()
