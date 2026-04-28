import numpy as np
import adi
import tensorflow as tf
from tensorflow import keras
import time
from scipy import signal

# ============================================================
# 1. SETUP & PARAMETERS
# ============================================================
FS = 1_000_000
CENTER_FREQ = 915_000_000
FRAME_LEN = 1024
MOD_CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM", "GMSK", "OQPSK", "AM-DSB-WC", "FM"]
CONFIDENCE_THRESHOLD = 0.7  # 70% threshold for 'Autre' classification

# Load the brain
print("[LOAD] Loading trained model...")
model = keras.models.load_model(r"C:\Users\hayet\radioML 2018\pluto_final_classifier.h5", compile=False)

# Setup Hardware
sdr = adi.Pluto(uri="ip:192.168.2.1")
sdr.sample_rate = int(FS)
sdr.rx_lo = int(CENTER_FREQ)
sdr.tx_lo = int(CENTER_FREQ)
sdr.tx_cyclic_buffer = True  # CRITICAL: Keeps TX running while we RX
sdr.rx_buffer_size = FRAME_LEN
sdr.tx_hardwaregain_chan0 = -15.0
sdr.rx_hardwaregain_chan0 = 25.0

# ============================================================
# 2. SIGNAL GENERATION HELPER (MATCHES TRAINING)
# ============================================================
def get_symbols(mod_type, num_symbols):
    """Generate constellation symbols (EXACT MATCH to training)."""
    if mod_type == "BPSK":
        syms = np.array([-1, 1])
        indices = np.random.randint(0, 2, num_symbols)
    elif mod_type == "QPSK":
        syms = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        indices = np.random.randint(0, 4, num_symbols)
    elif mod_type == "8PSK":
        phases = np.arange(8) * (2 * np.pi / 8)
        syms = np.exp(1j * phases)
        indices = np.random.randint(0, 8, num_symbols)
    elif mod_type == "16QAM":
        x = np.array([-3, -1, 1, 3])
        X, Y = np.meshgrid(x, x)
        syms = (X.flatten() + 1j * Y.flatten()) / np.sqrt(10)
        indices = np.random.randint(0, 16, num_symbols)
    elif mod_type == "64QAM":
        x = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        X, Y = np.meshgrid(x, x)
        syms = (X.flatten() + 1j * Y.flatten()) / np.sqrt(42)
        indices = np.random.randint(0, 64, num_symbols)
    elif mod_type == "256QAM":
        x = np.array([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15])
        X, Y = np.meshgrid(x, x)
        syms = (X.flatten() + 1j * Y.flatten()) / np.sqrt(170)
        indices = np.random.randint(0, 256, num_symbols)
    elif mod_type == "GMSK":
        syms = np.exp(1j * np.cumsum(np.random.randint(0, 2, num_symbols) * np.pi))
    elif mod_type == "OQPSK":
        I = np.array([-1, 1])[np.random.randint(0, 2, num_symbols)]
        Q = np.array([-1, 1])[np.random.randint(0, 2, num_symbols)]
        syms = (I + 1j * Q) / np.sqrt(2)
    elif mod_type == "AM-DSB-WC":
        t = np.arange(num_symbols) / FS
        message = np.sin(2*np.pi*0.01*t)
        syms = (1 + 0.8 * message) * np.exp(1j * 2*np.pi*0.1*t)
    elif mod_type == "FM":
        t = np.arange(num_symbols) / FS
        message = np.sin(2*np.pi*0.01*t)
        phase = 2*np.pi*50 * np.cumsum(message) / FS
        syms = np.exp(1j * phase)
    else:
        raise ValueError(f"Unknown modulation: {mod_type}")
    
    if hasattr(syms, '__len__') and len(syms) > 1 and 'indices' in locals():
        return syms[indices].astype(complex)
    return syms.astype(complex)

def apply_tx_filter(symbols, oversample=8):
    """Apply RRC filter (EXACT MATCH to training)."""
    oversampled = np.zeros(len(symbols) * oversample, dtype=complex)
    oversampled[::oversample] = symbols
    num_taps = 101
    beta = 0.35
    t = np.arange(-num_taps//2, num_taps//2 + 1) / oversample
    rrc = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2 + 1e-9)
    tx_signal = np.convolve(oversampled, rrc, mode='same')
    return (tx_signal / np.max(np.abs(tx_signal))) * 0.3

def generate_test_sig(mod_type):
    """Generate test signal using exact training methodology."""
    raw_syms = get_symbols(mod_type, num_symbols=4096)
    tx_sig = apply_tx_filter(raw_syms)
    return tx_sig

# ============================================================
# 3. THE EXPERIMENT
# ============================================================
# Get user input
print(f"\nAvailable modulations: {MOD_CLASSES}")
choice = input("Enter modulation to test (e.g., QPSK): ").upper()

if choice in MOD_CLASSES:
    print(f"\n[1] Generating {choice} signal...")
    sig = generate_test_sig(choice)
    
    print(f"[2] Starting continuous TX...")
    sdr.tx(sig * 10000)
    time.sleep(1) # Let the hardware stabilize
    
    print(f"[3] Capturing LIVE signal from RX port...")
    rx_sig = sdr.rx()
    
    # Pre-process (EXACT MATCH to training pipeline)
    rx_sig = rx_sig - np.mean(rx_sig)
    
    # Frequency correction (from training)
    psd = np.abs(np.fft.fftshift(np.fft.fft(rx_sig)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(rx_sig), d=1/FS))
    peak_freq = freqs[np.argmax(psd)]
    t = np.arange(len(rx_sig)) / FS
    rx_sig = rx_sig * np.exp(-1j * 2 * np.pi * peak_freq * t)
    
    # Normalize
    rx_sig = rx_sig / (np.sqrt(np.mean(np.abs(rx_sig)**2)) + 1e-9)
    
    # Reshape for AI
    I, Q = np.real(rx_sig), np.imag(rx_sig)
    input_data = np.stack([I, Q], axis=1).reshape(1, 1024, 2)
    
    # Predict
    print(f"[4] Running AI Prediction...")
    preds = model.predict(input_data, verbose=0)[0]
    max_confidence = np.max(preds)
    result_idx = np.argmax(preds)
    
    # Determine output: use model's guess if confidence high, else "Autre"
    if max_confidence >= CONFIDENCE_THRESHOLD:
        detected_class = MOD_CLASSES[result_idx]
        confidence_display = f"{max_confidence*100:.2f}%"
    else:
        detected_class = "Autre (Unidentified)"
        confidence_display = f"{max_confidence*100:.2f}% (Below {CONFIDENCE_THRESHOLD*100:.0f}% threshold)"
    
    print("\n" + "="*50)
    print(f"REALITY:     {choice}")
    print(f"AI DETECTED: {detected_class}")
    print(f"CONFIDENCE:  {confidence_display}")
    if max_confidence >= CONFIDENCE_THRESHOLD:
        print(f"\n✅ HIGH CONFIDENCE - Signal identified")
    else:
        print(f"\n⚠️  LOW CONFIDENCE - Signal rejected (noise/unknown)")
    print("="*50)

    sdr.tx_destroy_buffer()
else:
    print("Invalid modulation choice.")