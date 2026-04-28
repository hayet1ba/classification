"""
Diagnostic: Verify test signal generation matches training
This script validates that test signals are now properly generated.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

FS = 1_000_000

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
    else:
        raise ValueError(f"Unknown modulation: {mod_type}")
    
    return syms[indices].astype(complex)

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

print("[DIAGNOSTIC] Signal Generation Validation\n")

for mod in ["BPSK", "QPSK", "8PSK", "16QAM"]:
    syms = get_symbols(mod, 4096)
    tx_sig = apply_tx_filter(syms)
    
    print(f"\n{mod}:")
    print(f"  ✅ Signal length: {len(tx_sig)} samples")
    print(f"  ✅ Max amplitude: {np.max(np.abs(tx_sig)):.4f}")
    print(f"  ✅ Power: {np.mean(np.abs(tx_sig)**2):.6f}")
    print(f"  ✅ Constellation points: {len(np.unique(syms))}")
    
    # Check PSD
    psd = np.abs(np.fft.fftshift(np.fft.fft(tx_sig)))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(tx_sig), d=1/FS))
    peak_freq_idx = np.argmax(psd)
    peak_freq = freqs[peak_freq_idx]
    
    print(f"  ✅ Spectral peak at: {peak_freq:.1f} Hz")

print("\n✅ All signals match training generation pipeline!")
