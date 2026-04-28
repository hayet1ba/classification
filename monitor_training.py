"""
Monitor PlutoSDR Model Training
This script checks if training has completed and the model is ready.
"""
import os
import time
from pathlib import Path

model_path = r"C:\Users\hayet\radioML 2018\pluto_final_classifier.h5"

print("[MONITOR] Checking training status...\n")

while True:
    if os.path.exists(model_path):
        file_size = os.path.getsize(model_path)
        mod_time = os.path.getmtime(model_path)
        current_time = time.time()
        age_seconds = current_time - mod_time
        
        print(f"✅ Model found: {model_path}")
        print(f"   Size: {file_size / 1024 / 1024:.2f} MB")
        print(f"   Last modified: {age_seconds:.0f} seconds ago")
        
        if age_seconds < 30:
            print("\n[INFO] Training just completed! Model is fresh.")
        break
    else:
        print("[WAITING] Model not yet saved. Training in progress...")
        print("         Checking again in 10 seconds...\n")
        time.sleep(10)

print("\n✅ Ready to run inference!")
