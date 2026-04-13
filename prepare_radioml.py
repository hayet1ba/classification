import json
import h5py
import numpy as np
from pathlib import Path

# -----------------------------
# 1. Paths and class definitions
# -----------------------------

# Adjust these to match your files
H5_PATH = r"GOLD_XYZ_OSC.0001_1024.hdf5"
CLASSES_FIXED_JSON = r"classes-fixed.json"

# Output file (preprocessed dataset)
OUT_H5_PATH = r"RML2018_prepared.h5"

# Modulation types to remove from the dataset
MODULATIONS_TO_DELETE = {
    "OOK",
    "4ASK",
    "8ASK",
    "16PSK",
    "32PSK",
    "16APSK",
    "32APSK",
    "64APSK",
    "128APSK",
    "32QAM",
    "128QAM",
    "AM-SSB-WC",
    "AM-SSB-SC",
    "AM-DSB-SC",
}


# -----------------------------
# 2. Load classes-fixed.json
# -----------------------------

with open(CLASSES_FIXED_JSON, "r", encoding="utf-8") as f:
    classes_fixed = json.load(f)  # list of 24 modulation names

num_classes = len(classes_fixed)
print("classes_fixed:", classes_fixed)
print("num_classes:", num_classes)


# -----------------------------
# 3. Load original HDF5 dataset
# -----------------------------

with h5py.File(H5_PATH, "r") as f:
    # Adjust keys here if your file uses different names
    X_raw = f["X"][:]  # expected shape (N, 2, 1024)
    Y_raw = f["Y"][:]  # expected shape (N, num_classes) one-hot
    Z_raw = f["Z"][:] if "Z" in f else None  # SNR (optional)

# Delete samples whose modulation type is in MODULATIONS_TO_DELETE
delete_class_indices = [
    i for i, mod_name in enumerate(classes_fixed) if mod_name in MODULATIONS_TO_DELETE
]
if delete_class_indices:
    y_idx_all = np.argmax(Y_raw, axis=1)
    keep_mask = ~np.isin(y_idx_all, delete_class_indices)
    removed_count = int((~keep_mask).sum())

    X_raw = X_raw[keep_mask]
    Y_raw = Y_raw[keep_mask]
    if Z_raw is not None:
        Z_raw = Z_raw[keep_mask]

    print(f"Removed {removed_count} samples for excluded modulation types.")

print("X_raw shape:", X_raw.shape)
print("Y_raw shape:", Y_raw.shape)
if Z_raw is not None:
    print("Z_raw shape:", Z_raw.shape)

N = X_raw.shape[0]


# -----------------------------
# 4. Clean / standardize labels
# -----------------------------

# y_mod: integer labels 0..23 (modulation index)
# if Y_raw is one-hot, argmax over axis=1 gives class index
y_mod = np.argmax(Y_raw, axis=1).astype(np.int64)   # shape (N,)

# Optional: verify that number of classes matches
assert Y_raw.shape[1] == num_classes, "Mismatch between Y columns and classes-fixed.json"

# y_mod_onehot: can just be Y_raw if already one-hot
y_mod_onehot = Y_raw.astype(np.float32)

# y_snr: SNR labels (flatten to 1D if present)
if Z_raw is not None:
    y_snr = Z_raw.reshape(-1).astype(np.float32)    # shape (N,)
else:
    y_snr = None

print("y_mod shape:", y_mod.shape)
if y_snr is not None:
    print("y_snr shape:", y_snr.shape)


# -----------------------------
# 5 & 6. Preprocess and Save in CHUNKS (Memory Friendly)
# -----------------------------
out_path = Path(OUT_H5_PATH)
if out_path.exists():
    out_path.unlink()

with h5py.File(out_path, "w") as f_out:
    # Prepare the empty datasets first
    f_out.create_dataset("X", shape=(N, 1024, 2), dtype="float32", compression="gzip")
    f_out.create_dataset("y_mod", data=y_mod, compression="gzip")
    f_out.create_dataset("y_mod_onehot", data=y_mod_onehot, compression="gzip")
    if y_snr is not None:
        f_out.create_dataset("y_snr", data=y_snr, compression="gzip")
    f_out.attrs["classes"] = np.array(classes_fixed, dtype="S")

    # Process in chunks of 50,000 signals to save RAM
    chunk_size = 50000
    print(f"Starting chunked processing (Total signals: {N})...")
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        
        # Take a slice of the data
        X_chunk = X_raw[i:end].astype(np.float32)
        
        # Fix axis if needed (N, 2, 1024) -> (N, 1024, 2)
        if X_chunk.shape[1] == 2:
            X_chunk = np.transpose(X_chunk, (0, 2, 1))
            
        # Normalize this chunk
        rms = np.sqrt(np.mean(X_chunk**2, axis=(1, 2), keepdims=True))
        X_chunk = X_chunk / (rms + 1e-12)
        
        # Write this chunk to the file
        f_out["X"][i:end] = X_chunk
        
        print(f"  Processed {end}/{N} signals...")

print("Success! Saved prepared dataset to:", OUT_H5_PATH)