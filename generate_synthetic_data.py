"""
Generate Synthetic Training Data for ALL 10 Modulation Classes
This creates a complete dataset without needing PlutoSDR hardware
"""
import numpy as np
from scipy import signal
import os

print("[SYNTHETIC DATA] Generating training data for all 10 modulation classes\n")

# Parameters
FS = 1_000_000
FRAME_LEN = 1024
FRAMES_PER_MOD = 1000  # 1000 frames per class
NUM_SAMPLES = 1024 * 64

MOD_CLASSES = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM", "GMSK", "OQPSK", "AM-DSB-WC", "FM"]

def get_symbols(mod_type, num_symbols):
    """Generate constellation symbols."""
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
        # GMSK: continuous phase FSK
        syms = np.exp(1j * np.cumsum(np.random.randint(0, 2, num_symbols) * np.pi))
    elif mod_type == "OQPSK":
        # OQPSK: offset QPSK
        I = np.array([-1, 1])[np.random.randint(0, 2, num_symbols)]
        Q = np.array([-1, 1])[np.random.randint(0, 2, num_symbols)]
        syms = (I + 1j * Q) / np.sqrt(2)
    elif mod_type == "AM-DSB-WC":
        # AM with carrier
        t = np.arange(num_symbols) / FS
        message = np.sin(2*np.pi*0.01*t)
        syms = (1 + 0.8 * message) * np.exp(1j * 2*np.pi*0.1*t)
    elif mod_type == "FM":
        # Frequency modulation
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
    """Apply RRC pulse shaping filter."""
    oversampled = np.zeros(len(symbols) * oversample, dtype=complex)
    oversampled[::oversample] = symbols
    num_taps = 101
    beta = 0.35
    t = np.arange(-num_taps//2, num_taps//2 + 1) / oversample
    rrc = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2 + 1e-9)
    tx_signal = np.convolve(oversampled, rrc, mode='same')
    return (tx_signal / np.max(np.abs(tx_signal))) * 0.3

def simulate_channel(tx_sig):
    """Simulate Pluto RX processing."""
    # Add tiny amount of noise
    noise = np.random.normal(0, 0.01, len(tx_sig)) + 1j*np.random.normal(0, 0.01, len(tx_sig))
    rx_sig = tx_sig + noise
    
    # Remove DC
    rx_sig = rx_sig - np.mean(rx_sig)
    
    # Frequency correction
    psd = np.abs(np.fft.fftshift(np.fft.fft(rx_sig)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(rx_sig), d=1/FS))
    peak_freq = freqs[np.argmax(psd)]
    t = np.arange(len(rx_sig)) / FS
    rx_sig = rx_sig * np.exp(-1j * 2 * np.pi * peak_freq * t)
    
    # Normalize
    rx_sig = rx_sig / (np.sqrt(np.mean(np.abs(rx_sig)**2)) + 1e-9)
    
    return rx_sig

# Generate data
all_frames = []
all_labels = []

for mod in MOD_CLASSES:
    print(f"[GENERATING] {mod}...")
    
    for frame_num in range(FRAMES_PER_MOD):
        # Generate symbols
        raw_syms = get_symbols(mod, num_symbols=4096)
        tx_sig = apply_tx_filter(raw_syms)
        
        # Simulate channel
        rx_sig = simulate_channel(tx_sig)
        
        # Split into frames
        num_slices = len(rx_sig) // FRAME_LEN
        for i in range(num_slices):
            frame = rx_sig[i*FRAME_LEN : (i+1)*FRAME_LEN]
            
            if np.max(np.abs(frame)) > 0.05:
                I, Q = np.real(frame), np.imag(frame)
                all_frames.append(np.stack([I, Q], axis=1))
                all_labels.append(mod)
    
    print(f"  ✅ {mod}: {FRAMES_PER_MOD} base frames generated")

print(f"\n[SUMMARY]")
print(f"  Total frames: {len(all_frames)}")
print(f"  Classes: {len(set(all_labels))}")
print(f"  Class distribution:")
for cls in MOD_CLASSES:
    count = sum(1 for x in all_labels if x == cls)
    print(f"    - {cls}: {count} frames")

# Save
print(f"\n[SAVING] Writing to disk...")
np.save("pluto_X_train_synthetic.npy", np.array(all_frames).astype(np.float32))
np.save("pluto_Y_train_synthetic.npy", np.array(all_labels))

print(f"✅ Synthetic dataset saved!")
print(f"   - pluto_X_train_synthetic.npy ({len(all_frames)} frames)")
print(f"   - pluto_Y_train_synthetic.npy ({len(all_labels)} labels)")

# Show file sizes
x_size = os.path.getsize("pluto_X_train_synthetic.npy") / 1024 / 1024
y_size = os.path.getsize("pluto_Y_train_synthetic.npy") / 1024 / 1024
print(f"   - X size: {x_size:.2f} MB")
print(f"   - Y size: {y_size:.2f} MB")
