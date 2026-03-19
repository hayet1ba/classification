import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


import tensorflow as tf
from tensorflow.keras import layers, models

# Path to your PREPARED file
# Default to your workspace local file if present
local_default = os.path.join(os.path.dirname(__file__), 'RML2018_prepared.h5')
PREPARED_H5 = local_default if os.path.exists(local_default) else r"C:\Users\hayet\Downloads\radioML 2018\RML2018_prepared.h5"

if not os.path.exists(PREPARED_H5):
    raise FileNotFoundError(f"Prepared HDF5 file not found: {PREPARED_H5}.\nPlease set PREPARED_H5 to correct path")

# Known classes from classes-fixed.txt
CLASSES = ['OOK','4ASK','8ASK','BPSK','QPSK','8PSK','16PSK','32PSK','16APSK','32APSK','64APSK','128APSK','16QAM','32QAM','64QAM','128QAM','256QAM','AM-SSB-WC','AM-SSB-SC','AM-DSB-WC','AM-DSB-SC','FM','GMSK','OQPSK']


def augment_iq_features(X):
    """Add magnitude and phase channels to I/Q input."""
    complex_signal = X[..., 0] + 1j * X[..., 1]
    mag = np.abs(complex_signal)
    phase = np.angle(complex_signal)

    mag = (mag - np.mean(mag, axis=1, keepdims=True)) / (np.std(mag, axis=1, keepdims=True) + 1e-8)
    phase = (phase - np.mean(phase, axis=1, keepdims=True)) / (np.std(phase, axis=1, keepdims=True) + 1e-8)

    extra = np.stack([mag, phase], axis=2)
    return np.concatenate([X, extra], axis=2)


def load_and_split_data(file_path, min_snr=None, subset=None):
    print(f"Opening {file_path}...")
    with h5py.File(file_path, "r") as f:
        snr = f["y_snr"][:]
        Y = f["y_mod_onehot"][:]

        if min_snr is not None:
            print(f"Filtering for SNR >= {min_snr} dB...")
            keep = np.where(snr >= min_snr)[0]
        else:
            print("Using full SNR range for robustness...")
            keep = np.arange(len(snr))

        if subset is not None and 0 < subset < 1:
            subset_n = max(1, int(len(keep) * subset))
            rgen = np.random.RandomState(42)
            keep = rgen.choice(keep, size=subset_n, replace=False)
            keep = np.sort(keep)
            print(f"Subsetting to {subset*100:.1f}% ({subset_n} samples)")

        print(f"Loading {len(keep)} samples into memory...")
        X = f["X"][keep]
        Y = Y[keep]

    print("Augmenting features (I/Q + magnitude + phase)...")
    X = augment_iq_features(X)

    print("Splitting data into Train/Val/Test (80/10/10)...")
    train_idx, temp_idx = train_test_split(
        np.arange(len(X)), test_size=0.20, random_state=42, stratify=np.argmax(Y, axis=1)
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=42, stratify=np.argmax(Y[temp_idx], axis=1)
    )

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    Y_train, Y_val, Y_test = Y[train_idx], Y[val_idx], Y[test_idx]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def residual_block(x, filters, kernel_size=3, strides=1):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same', strides=strides, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv1D(filters, kernel_size, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if shortcut.shape[-1] != filters or strides != 1:
        shortcut = layers.Conv1D(filters, 1, padding='same', strides=strides)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.add([x, shortcut])
    x = layers.ReLU()(x)
    x = layers.SpatialDropout1D(0.2)(x)
    return x


def build_model(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(64, 3, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.SpatialDropout1D(0.2)(x)

    x = residual_block(x, 128, kernel_size=3, strides=1)
    x = layers.MaxPooling1D(2)(x)

    x = residual_block(x, 256, kernel_size=3, strides=1)
    x = layers.MaxPooling1D(2)(x)

    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train modulation classifier on prepared RadioML dataset.')
    parser.add_argument('--min-snr', type=float, default=None, help='Minimum SNR filter value')
    parser.add_argument('--subset', type=float, default=1.0, help='Fraction of the dataset to use (0 < subset <= 1)')
    args = parser.parse_args()

    X_train, X_val, X_test, Y_train, Y_val, Y_test = load_and_split_data(
        PREPARED_H5, min_snr=args.min_snr, subset=args.subset if args.subset < 1.0 else None
    )

    print("\n--- Final Data Shapes ---")
    print(f"Training:   {X_train.shape} (80%)")
    print(f"Validation: {X_val.shape} (10%)")
    print(f"Testing:    {X_test.shape} (10%)")

    model = build_model(input_shape=X_train.shape[1:], num_classes=len(CLASSES))
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    print("\nStarting training with early stopping + LR schedule...")
    history = model.fit(
        X_train, Y_train,
        epochs=100,
        batch_size=128,
        validation_data=(X_val, Y_val),
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, Y_test, verbose=1)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title('Accuracy')

    plt.savefig('training_history.png', dpi=150)
    plt.close()
    print('Training curves saved to training_history.png')

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print('\nClassification report (full test):')
    print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(CLASSES))
    plt.xticks(tick_marks, CLASSES, rotation=90)
    plt.yticks(tick_marks, CLASSES)
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.close()
    print('Confusion matrix saved to confusion_matrix.png')

    selected = ['BPSK', 'QPSK', '16QAM', '256QAM']
    sel_idxs = [CLASSES.index(c) for c in selected]
    mask = np.isin(y_true, sel_idxs)
    if np.any(mask):
        y_true_small = y_true[mask]
        y_pred_small = y_pred[mask]
        print('\n4-modulation test metrics: BPSK, QPSK, 16QAM, 256QAM')
        print(classification_report(y_true_small, y_pred_small,
                                    target_names=selected, zero_division=0))
    else:
        print('No test samples for the selected four modulations in this split.')

    model.save('radio_modulation_model_improved.h5')
    print('\nModel saved as radio_modulation_model_improved.h5')


    
   