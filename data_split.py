import h5py
import numpy as np
from sklearn.model_selection import train_test_split


SPLIT_MODE = "stratified"  # "stratified" | "snr_shift"

# Seuils SNR (mode snr_shift uniquement)
S_TRAIN_MIN = 10
S_TEST_MAX = 10

PREP_FILE = "RML2018_prepared.h5"

with h5py.File(PREP_FILE, "r") as f:
    X = f["X"][:]
    y_mod = f["y_mod_onehot"][:]
    y_snr = f["y_snr"][:]
    classes = f.attrs["classes"]

print("Dataset loaded.")
print("Shapes:", X.shape, y_mod.shape, y_snr.shape)
print(f"SPLIT_MODE = {SPLIT_MODE!r}")

if SPLIT_MODE == "snr_shift":
    train_mask = y_snr >= S_TRAIN_MIN
    test_mask  = y_snr < S_TEST_MAX

    X_train_full  = X[train_mask]
    y_train_full  = y_mod[train_mask]
    snr_train_full = y_snr[train_mask]   # SNR du pool train

    X_test  = X[test_mask]
    y_test  = y_mod[test_mask]
    snr_test = y_snr[test_mask]          # SNR du test

    print(
        f"SNR shift: train/val pool SNR >= {S_TRAIN_MIN} "
        f"(min..max ≈ {float(snr_train_full.min()):.1f} .. {float(snr_train_full.max()):.1f}), "
        f"test SNR < {S_TEST_MAX} "
        f"(≈ {float(snr_test.min()):.1f} .. {float(snr_test.max()):.1f})."
    )
    print("Train pool:", X_train_full.shape, "Test:", X_test.shape)

    X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
        X_train_full, y_train_full, snr_train_full,
        test_size=0.15, random_state=42, shuffle=True
    )

elif SPLIT_MODE == "stratified":
    y_lab = np.argmax(y_mod, axis=1)

    # 1er split : train+val / test (on split aussi y_snr)
    X_tv, X_test, y_tv, y_test, snr_tv, snr_test = train_test_split(
        X, y_mod, y_snr,
        test_size=0.2, random_state=42, shuffle=True, stratify=y_lab
    )

    # 2e split : train / val
    y_lab_tv = np.argmax(y_tv, axis=1)
    X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
        X_tv, y_tv, snr_tv,
        test_size=0.15 / 0.80,
        random_state=42,
        shuffle=True,
        stratify=y_lab_tv,
    )

    print(
        f"Stratified (tous SNR): SNR global min..max = "
        f"{float(y_snr.min()):.1f} .. {float(y_snr.max()):.1f} "
        "(même distribution SNR dans train / val / test, à stratification près)."
    )

else:
    raise ValueError(f"Unknown SPLIT_MODE: {SPLIT_MODE}")

print("Final splits:")
print("X_train:", X_train.shape, "| SNR train min/max:", snr_train.min(), snr_train.max())
print("X_val  :", X_val.shape,   "| SNR val   min/max:", snr_val.min(),   snr_val.max())
print("X_test :", X_test.shape,  "| SNR test  min/max:", snr_test.min(),  snr_test.max())

OUT_FILE = "RML2018_snr_split.h5"

with h5py.File(OUT_FILE, "w") as f_out:
    f_out.create_dataset("X_train",     data=X_train,    compression="gzip")
    f_out.create_dataset("y_train",     data=y_train,    compression="gzip")
    f_out.create_dataset("y_snr_train", data=snr_train,  compression="gzip")  # NOUVEAU
    f_out.create_dataset("X_val",       data=X_val,      compression="gzip")
    f_out.create_dataset("y_val",       data=y_val,      compression="gzip")
    f_out.create_dataset("y_snr_val",   data=snr_val,    compression="gzip")  # NOUVEAU
    f_out.create_dataset("X_test",      data=X_test,     compression="gzip")
    f_out.create_dataset("y_test",      data=y_test,     compression="gzip")
    f_out.create_dataset("y_snr_test",  data=snr_test,   compression="gzip")  # NOUVEAU
    f_out.create_dataset("classes",     data=classes)
    f_out.attrs["split_mode"] = SPLIT_MODE.encode("utf-8")

print("Done! Saved to", OUT_FILE)
print("Clés sauvegardées : X_train, y_train, y_snr_train, X_val, y_val, y_snr_val, X_test, y_test, y_snr_test, classes")