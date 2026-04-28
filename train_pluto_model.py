import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

# GPU Memory Fix
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# 1. LOAD NEW DATA
X = np.load("pluto_X_train.npy")
Y_text = np.load("pluto_Y_train.npy")

# Map your 10 classes (standardized names)
MOD_CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM", "GMSK", "OQPSK", "AM-DSB-WC", "FM"]

# Handle legacy class name mapping (QAM16 -> 16QAM)
LEGACY_MAPPING = {"QAM16": "16QAM"}
Y_text_mapped = np.array([LEGACY_MAPPING.get(label, label) for label in Y_text])

# Verify and filter labels
unique_labels = np.unique(Y_text_mapped)
print(f"[INFO] Training data labels: {unique_labels}")
print(f"[INFO] Target classes: {MOD_CLASSES}")

mask = np.array([label in MOD_CLASSES for label in Y_text_mapped])
X = X[mask]
Y_text_mapped = Y_text_mapped[mask]

if len(np.unique(Y_text_mapped)) < len(MOD_CLASSES):
    print(f"[WARNING] Only {len(np.unique(Y_text_mapped))}/{len(MOD_CLASSES)} classes available for training.\n")

Y_indices = np.array([MOD_CLASSES.index(label) for label in Y_text_mapped])
Y_onehot = keras.utils.to_categorical(Y_indices, num_classes=len(MOD_CLASSES))

X_train, X_val, Y_train, Y_val = train_test_split(X, Y_onehot, test_size=0.2, random_state=42)

# 2. MODIFY MODEL FOR 10 CLASSES
# We need to replace the top layer because the original model had 24 classes
base_model = keras.models.load_model(r"C:\Users\hayet\radioML 2018\best_model.h5", compile=False)

# Reconstruct the model to output 10 classes instead of 24
inputs = base_model.input
# Get the output of the layer before the final dense layer
x = base_model.layers[-2].output 
outputs = keras.layers.Dense(len(MOD_CLASSES), activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# 3. COMPILE & TRAIN
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\n[TRAIN] Training on clean PlutoSDR data...")
model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=15,
    batch_size=32,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

model.save(r"C:\Users\hayet\radioML 2018\pluto_final_classifier.h5")
print("✅ Final Model Saved!")
