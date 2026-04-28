import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv1D,
    BatchNormalization,
    Dropout,
    GlobalAveragePooling1D,
    Dense,
    Input,
    Add,
    Activation,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.utils import Sequence
import tensorflow as tf


class BalancedBatchSequence(Sequence):
    """Chaque batch contient le même nombre d'exemples par classe."""

    def __init__(self, X, y, batch_size, seed=42):
        super().__init__()
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y)
        self.num_classes = y.shape[1]
        if batch_size % self.num_classes != 0:
            raise ValueError(
                f"batch_size ({batch_size}) doit être un multiple de num_classes ({self.num_classes}) "
                "pour des batches équilibrés."
            )
        self.per_class = batch_size // self.num_classes
        self.batch_size = batch_size
        self.y_int = np.argmax(self.y, axis=1)
        self.rng = np.random.default_rng(seed)
        self.class_rows = [np.where(self.y_int == c)[0] for c in range(self.num_classes)]
        counts = [len(r) for r in self.class_rows]
        if any(n < self.per_class for n in counts):
            raise ValueError(
                f"Chaque classe doit avoir au moins {self.per_class} échantillons; comptes: {counts}"
            )
        self._n_batches = min(n // self.per_class for n in counts)
        self.on_epoch_end()

    def on_epoch_end(self):
        self.rows_by_class = [
            self.rng.permutation(rows) for rows in self.class_rows
        ]

    def __len__(self):
        return self._n_batches

    def __getitem__(self, i):
        parts_x, parts_y = [], []
        for c in range(self.num_classes):
            sl = self.rows_by_class[c][
                i * self.per_class : (i + 1) * self.per_class
            ]
            parts_x.append(self.X[sl])
            parts_y.append(self.y[sl])
        X_batch = np.concatenate(parts_x, axis=0)
        y_batch = np.concatenate(parts_y, axis=0)
        p = self.rng.permutation(self.batch_size)
        return X_batch[p], y_batch[p]


# ============================================================
# 1. CONFIG GPU
# ============================================================

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU actif :", gpus[0].name)
    except RuntimeError as e:
        print(e)

INFER_BATCH = 16

# ============================================================
# 2. CHARGEMENT DATASET + FILTRAGE SNR >= 0 dB
# ============================================================

DATASET_PATH = r"C:\Users\hayet\radioML 2018\RML2018_snr_split.h5"
SNR_MIN = 0  # FIX 1 : on garde uniquement SNR >= 0 dB

print("\n--- Chargement du dataset ---")
with h5py.File(DATASET_PATH, "r") as f:
    X_train_raw = f["X_train"][:]
    y_train_raw = f["y_train"][:]
    X_val_raw   = f["X_val"][:]
    y_val_raw   = f["y_val"][:]
    X_test_raw  = f["X_test"][:]
    y_test_raw  = f["y_test"][:]
    y_snr_train = f["y_snr"][:][:len(f["X_train"])] if "y_snr" in f else None
    all_classes_raw = [c.decode() for c in f["classes"][:]]
    _sm = f.attrs.get("split_mode", b"")
    split_mode_h5 = _sm.decode("utf-8") if isinstance(_sm, (bytes, bytearray)) else str(_sm)

    # Charge les SNR pour chaque split si disponibles séparément
    y_snr_train = f["y_snr_train"][:] if "y_snr_train" in f else None
    y_snr_val   = f["y_snr_val"][:]   if "y_snr_val"   in f else None
    y_snr_test  = f["y_snr_test"][:]  if "y_snr_test"  in f else None

print("Données chargées")
print("Classes d'origine :", all_classes_raw)
if split_mode_h5:
    print("Split mode :", split_mode_h5)

# Filtrage SNR >= 0 dB
if y_snr_train is not None:
    mask_train = y_snr_train >= SNR_MIN
    mask_val   = y_snr_val   >= SNR_MIN
    mask_test  = y_snr_test  >= SNR_MIN
    X_train_raw = X_train_raw[mask_train]
    y_train_raw = y_train_raw[mask_train]
    X_val_raw   = X_val_raw[mask_val]
    y_val_raw   = y_val_raw[mask_val]
    X_test_raw  = X_test_raw[mask_test]
    y_test_raw  = y_test_raw[mask_test]
    print(f"Filtrage SNR >= {SNR_MIN} dB appliqué :")
    print(f"  train={X_train_raw.shape[0]}  val={X_val_raw.shape[0]}  test={X_test_raw.shape[0]}")
else:
    print("Attention : clés y_snr_train/val/test non trouvées dans le fichier H5 — filtrage SNR ignoré.")
    print("Vérifie les clés disponibles dans ton fichier H5.")


# ============================================================
# 3. COLONNES DE CLASSES ACTIVES
# ============================================================

y_all_stacked = np.vstack([y_train_raw, y_val_raw, y_test_raw])
present_indices = np.where(np.sum(y_all_stacked, axis=0) > 0)[0]

classes = [all_classes_raw[i] for i in present_indices]
num_classes = len(classes)

print("\nColonnes one-hot actives :", classes)
print("num_classes :", num_classes)

y_train_full = y_train_raw[:, present_indices]
y_val_full   = y_val_raw[:, present_indices]
y_test_full  = y_test_raw[:, present_indices]


# ============================================================
# 4. FONCTIONS DE SOUS-ÉCHANTILLONNAGE
# ============================================================

def stratified_subsample(X, y_onehot, max_n, random_state=42):
    n = X.shape[0]
    if n <= max_n:
        return X.astype(np.float32), y_onehot
    y_int = np.argmax(y_onehot, axis=1)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=max_n, random_state=random_state)
    idx, _ = next(sss.split(np.zeros((n, 1)), y_int))
    return X[idx].astype(np.float32), y_onehot[idx]


def balanced_equal_sample(X, y_onehot, samples_per_class, random_state=42):
    rng = np.random.default_rng(random_state)
    X = np.asarray(X)
    y_onehot = np.asarray(y_onehot)
    y_int = np.argmax(y_onehot, axis=1)
    C = y_onehot.shape[1]
    counts = [int(np.sum(y_int == c)) for c in range(C)]
    if any(n == 0 for n in counts):
        raise ValueError(f"Classe sans échantillon : comptes={counts}")
    m = min(samples_per_class, min(counts))
    parts_x, parts_y = [], []
    for c in range(C):
        idx = np.where(y_int == c)[0]
        chosen = rng.choice(idx, size=m, replace=False)
        parts_x.append(X[chosen])
        parts_y.append(y_onehot[chosen])
    X_out = np.concatenate(parts_x, axis=0).astype(np.float32)
    y_out = np.concatenate(parts_y, axis=0)
    perm = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


# ============================================================
# 5. RÉDUCTION POUR GPU
# ============================================================

MAX_TRAIN = 200000
MAX_VAL   = 20000
MAX_TEST  = 20000
SAMPLES_PER_CLASS_TRAIN = MAX_TRAIN // num_classes

X_train, y_train = balanced_equal_sample(
    X_train_raw, y_train_full, SAMPLES_PER_CLASS_TRAIN, random_state=42
)
X_val, y_val = stratified_subsample(
    X_val_raw, y_val_full, MAX_VAL, random_state=43
)
X_test, y_test = stratified_subsample(
    X_test_raw, y_test_full, MAX_TEST, random_state=44
)

BATCH_PER_CLASS = 2
batch_size = num_classes * BATCH_PER_CLASS

print("\nShape X_train :", X_train.shape)
print("Shape X_val   :", X_val.shape)
print("Shape X_test  :", X_test.shape)
print("batch_size    :", batch_size, f"({BATCH_PER_CLASS} par classe)")


# ============================================================
# 6. ARCHITECTURE ResNet-1D
# ============================================================

def residual_block(x, filters, kernel_size=3, dilation_rate=1, dropout=0.15, project=False):
    shortcut = x
    if project:
        shortcut = Conv1D(filters, 1, padding="same")(shortcut)
        shortcut = BatchNormalization()(shortcut)
    y = Conv1D(filters, kernel_size, padding="same", dilation_rate=dilation_rate)(x)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout)(y)
    y = Conv1D(filters, kernel_size, padding="same")(y)
    y = BatchNormalization()(y)
    y = Add()([shortcut, y])
    y = Activation("relu")(y)
    y = Dropout(dropout)(y)
    return y


def build_resnet1d_mod(input_shape=(1024, 2), num_classes=10):
    inp = Input(shape=input_shape)
    x = Conv1D(64, 7, padding="same")(inp)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.1)(x)

    x = residual_block(x, 64,  kernel_size=3, dropout=0.12, project=False)
    x = residual_block(x, 128, kernel_size=3, dropout=0.15, project=True)
    x = residual_block(x, 256, kernel_size=3, dropout=0.18, project=True)
    x = residual_block(x, 256, kernel_size=3, dropout=0.2,  project=False)

    x = GlobalAveragePooling1D()(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.25)(x)  # FIX 4 : réduit de 0.4 à 0.25
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inp, out, name="resnet1d_mod")


model = build_resnet1d_mod(input_shape=(1024, 2), num_classes=num_classes)

# FIX 3 : label_smoothing = 0.0
loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.0)
model.compile(optimizer=Adam(1e-3), loss=loss_fn, metrics=["accuracy"])
model.summary()


# ============================================================
# 7. ENTRAÎNEMENT
# ============================================================

print("\n--- Entraînement ---")
train_seq = BalancedBatchSequence(X_train, y_train, batch_size=batch_size, seed=45)

callbacks = [
    # FIX 2 : tous les callbacks monitorent val_accuracy
    ReduceLROnPlateau(
        monitor="val_accuracy",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        mode="max",
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=12,
        restore_best_weights=True,
        mode="max",
        verbose=1,
    ),
    ModelCheckpoint(
        filepath="best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
]

history = model.fit(
    train_seq,
    validation_data=(X_val, y_val),
    epochs=10,
    callbacks=callbacks,
    verbose=1,
)


# ============================================================
# 8. ÉVALUATION TEST
# ============================================================

loss, acc = model.evaluate(X_test, y_test, batch_size=INFER_BATCH)
print(f"\nTest Accuracy (sans seuil) : {acc:.4f}")


# ============================================================
# 9. SEUIL OPTIMAL ROC SUR VALIDATION
# ============================================================

y_val_probs = model.predict(X_val, batch_size=INFER_BATCH)
y_val_true  = np.argmax(y_val, axis=1)
y_val_pred  = np.argmax(y_val_probs, axis=1)

confidences = np.max(y_val_probs, axis=1)
correct = (y_val_pred == y_val_true).astype(int)

fpr, tpr, thresholds = roc_curve(correct, confidences)
optimal_threshold = thresholds[np.argmax(tpr - fpr)]
print(f"\nSeuil optimal ROC : {optimal_threshold:.4f}")


# ============================================================
# 10. MATRICE DE CONFUSION
# ============================================================

y_test_probs = model.predict(X_test, batch_size=INFER_BATCH)
test_conf    = np.max(y_test_probs, axis=1)
test_pred    = np.argmax(y_test_probs, axis=1)

y_pred_final = np.copy(test_pred)
y_pred_final[test_conf < optimal_threshold] = num_classes  # classe "Autre"

classes_with_other = classes + ["Autre"]
y_true_labels = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_final, labels=range(len(classes_with_other)))

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap="Greens",
            xticklabels=classes_with_other,
            yticklabels=classes)
plt.title(f"Matrice de Confusion (Seuil={optimal_threshold:.2f}) — SNR >= {SNR_MIN} dB")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.tight_layout()
plt.show()


# ============================================================
# 11. SAUVEGARDE FINALE
# ============================================================

model.save("cnn1d_rml2018_corrected.h5")
print("\nModèle sauvegardé : cnn1d_rml2018_corrected.h5")