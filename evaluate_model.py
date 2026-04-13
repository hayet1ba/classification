import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_recall_fscore_support
from tensorflow.keras.models import load_model
import tensorflow as tf

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU active:", gpus[0].name)
    except RuntimeError as e:
        print(e)

# Load dataset
DATASET_PATH = r"C:\Users\hayet\radioML 2018\RML2018_snr_split.h5"
SNR_MIN = 0

print("\n--- Loading dataset ---")
with h5py.File(DATASET_PATH, "r") as f:
    X_train_raw = f["X_train"][:]
    y_train_raw = f["y_train"][:]
    X_val_raw = f["X_val"][:]
    y_val_raw = f["y_val"][:]
    X_test_raw = f["X_test"][:]
    y_test_raw = f["y_test"][:]
    all_classes_raw = [c.decode() for c in f["classes"][:]]

    # Load SNR for each split
    y_snr_train = f["y_snr_train"][:] if "y_snr_train" in f else None
    y_snr_val = f["y_snr_val"][:] if "y_snr_val" in f else None
    y_snr_test = f["y_snr_test"][:] if "y_snr_test" in f else None

print("Data loaded")
print("Original classes:", all_classes_raw)

# Filter SNR >= 0 dB
if y_snr_train is not None:
    mask_train = y_snr_train >= SNR_MIN
    mask_val = y_snr_val >= SNR_MIN
    mask_test = y_snr_test >= SNR_MIN
    X_train_raw = X_train_raw[mask_train]
    y_train_raw = y_train_raw[mask_train]
    X_val_raw = X_val_raw[mask_val]
    y_val_raw = y_val_raw[mask_val]
    X_test_raw = X_test_raw[mask_test]
    y_test_raw = y_test_raw[mask_test]
    print(f"SNR >= {SNR_MIN} dB filtering applied:")
    print(f"  train={X_train_raw.shape[0]}  val={X_val_raw.shape[0]}  test={X_test_raw.shape[0]}")
else:
    print("Warning: y_snr_train/val/test keys not found in H5 file — SNR filtering ignored.")

# Determine active classes
y_all_stacked = np.vstack([y_train_raw, y_val_raw, y_test_raw])
present_indices = np.where(np.sum(y_all_stacked, axis=0) > 0)[0]

classes = [all_classes_raw[i] for i in present_indices]
num_classes = len(classes)

print("\nActive one-hot columns:", classes)
print("num_classes:", num_classes)

y_test_full = y_test_raw[:, present_indices]

# Subsample test data if needed (same as training)
from sklearn.model_selection import StratifiedShuffleSplit

def stratified_subsample(X, y_onehot, max_n, random_state=42):
    n = X.shape[0]
    if n <= max_n:
        return X.astype(np.float32), y_onehot
    y_int = np.argmax(y_onehot, axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_n, random_state=random_state)
    idx, _ = next(sss.split(np.zeros((n, 1)), y_int))
    return X[idx].astype(np.float32), y_onehot[idx]

MAX_TEST = 20000
X_test, y_test = stratified_subsample(X_test_raw, y_test_full, MAX_TEST, random_state=44)

print("Test data shape:", X_test.shape)

# Load the best model
model_path = "best_model.h5"
print(f"\n--- Loading model from {model_path} ---")
model = load_model(model_path, compile=False)
# Recompile with current loss
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
model.compile(optimizer='adam', loss=loss_fn, metrics=["accuracy"])
model.summary()

# Evaluate on test set
print("\n--- Evaluating on test set ---")
loss, acc = model.evaluate(X_test, y_test, batch_size=16, verbose=1)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")

# Predictions
y_test_probs = model.predict(X_test, batch_size=16, verbose=1)
y_test_pred = np.argmax(y_test_probs, axis=1)
y_test_true = np.argmax(y_test, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_test_true, y_test_pred)
print("\nConfusion Matrix:")
print(cm)

# Classification Report
report = classification_report(y_test_true, y_test_pred, target_names=classes, digits=4)
print("\nClassification Report:")
print(report)

# Plot Confusion Matrix
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes, yticklabels=classes)
plt.title('Confusion Matrix - Best Model Evaluation')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig('confusion_matrix_best_model.png', dpi=300, bbox_inches='tight')
plt.show()

# Additional metrics
precision, recall, f1, support = precision_recall_fscore_support(y_test_true, y_test_pred, average=None)
print("\nPer-class metrics:")
for i, cls in enumerate(classes):
    print(f"{cls}: Precision={precision[i]:.4f}, Recall={recall[i]:.4f}, F1={f1[i]:.4f}, Support={support[i]}")

# Overall metrics
overall_precision = np.mean(precision)
overall_recall = np.mean(recall)
overall_f1 = np.mean(f1)
print(f"\nOverall Average Precision: {overall_precision:.4f}")
print(f"Overall Average Recall: {overall_recall:.4f}")
print(f"Overall Average F1-Score: {overall_f1:.4f}")

print("\nEvaluation complete. Confusion matrix saved as 'confusion_matrix_best_model.png'")