import numpy as np
import adi
import time
from scipy import signal

# ============================================================
# 1. PARAMETERS
# ============================================================
FS = 1_000_000
CENTER_FREQ = 915_000_000
BANDWIDTH = 1_000_000
NUM_SAMPLES = 1024 * 64 
FRAME_LEN = 1024
TARGET_MODS = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "256QAM", "GMSK", "OQPSK", "AM-DSB-WC", "FM"]
FRAMES_PER_MOD = 500

# ============================================================
# 2. SIGNAL GENERATION FUNCTIONS
# ============================================================
def get_symbols(mod_type, num_symbols):
    """Generate constellation symbols for modulation types."""
    if mod_type == "BPSK":
        syms = np.array([-1, 1])
        indices = np.random.randint(0, 2, num_symbols)
        return syms[indices].astype(complex)
    elif mod_type == "QPSK":
        syms = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        indices = np.random.randint(0, 4, num_symbols)
        return syms[indices].astype(complex)
    elif mod_type == "8PSK":
        phases = np.arange(8) * (2 * np.pi / 8)
        syms = np.exp(1j * phases)
        indices = np.random.randint(0, 8, num_symbols)
        return syms[indices].astype(complex)
    elif mod_type == "16QAM":
        x = np.array([-3, -1, 1, 3])
        X, Y = np.meshgrid(x, x)
        syms = (X.flatten() + 1j * Y.flatten()) / np.sqrt(10)
        indices = np.random.randint(0, 16, num_symbols)
        return syms[indices].astype(complex)
    elif mod_type == "64QAM":
        x = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        X, Y = np.meshgrid(x, x)
        syms = (X.flatten() + 1j * Y.flatten()) / np.sqrt(42)
        indices = np.random.randint(0, 64, num_symbols)
        return syms[indices].astype(complex)
    elif mod_type == "256QAM":
        x = np.array([-15, -13, -11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9, 11, 13, 15])
        X, Y = np.meshgrid(x, x)
        syms = (X.flatten() + 1j * Y.flatten()) / np.sqrt(170)
        indices = np.random.randint(0, 256, num_symbols)
        return syms[indices].astype(complex)
    elif mod_type == "GMSK":
        # Gaussian frequency shift keying (continuous phase)
        return np.exp(1j * np.cumsum(np.random.randint(0, 2, num_symbols) * np.pi)).astype(complex)
    elif mod_type == "OQPSK":
        # Offset QPSK (staggered I and Q)
        I = np.array([-1, 1])[np.random.randint(0, 2, num_symbols)]
        Q = np.array([-1, 1])[np.random.randint(0, 2, num_symbols)]
        return ((I + 1j * Q) / np.sqrt(2)).astype(complex)
    elif mod_type == "AM-DSB-WC":
        # AM Double-Sideband with Carrier
        t = np.arange(num_symbols) / FS
        message = np.sin(2*np.pi*0.01*t)
        carrier = np.exp(1j * 2*np.pi*0.1*t)
        return ((1 + 0.8 * message) * carrier).astype(complex)
    elif mod_type == "FM":
        # Frequency Modulation
        t = np.arange(num_symbols) / FS
        message = np.sin(2*np.pi*0.01*t)
        phase = 2*np.pi*50 * np.cumsum(message) / FS
        return np.exp(1j * phase).astype(complex)
    else:
        raise ValueError(f"Unknown modulation: {mod_type}")

def apply_tx_filter(symbols, oversample=8):
    oversampled = np.zeros(len(symbols) * oversample, dtype=complex)
    oversampled[::oversample] = symbols
    num_taps = 101
    beta = 0.35
    t = np.arange(-num_taps//2, num_taps//2 + 1) / oversample
    rrc = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t)**2 + 1e-9)
    tx_signal = np.convolve(oversampled, rrc, mode='same')
    return (tx_signal / np.max(np.abs(tx_signal))) * 0.3

# ============================================================
# 3. HARDWARE SETUP
# ============================================================
def setup_pluto():
    print("[PLUTO] Initializing for Data Collection...")
    try:
        sdr = adi.Pluto(uri="ip:192.168.2.1")
        sdr.sample_rate = int(FS)
        sdr.rx_lo = int(CENTER_FREQ)
        sdr.tx_lo = int(CENTER_FREQ)
        sdr.rx_rf_bandwidth = int(BANDWIDTH)
        sdr.tx_rf_bandwidth = int(BANDWIDTH)
        sdr.rx_buffer_size = NUM_SAMPLES
        sdr.tx_cyclic_buffer = True 
        sdr.gain_control_mode_chan0 = "manual"
        sdr.rx_hardwaregain_chan0 = 25.0
        sdr.tx_hardwaregain_chan0 = -20.0
        return sdr
    except Exception as e:
        print(f"❌ Connection Error: {e}")
        exit()

# ============================================================
# 4. DATA COLLECTION LOOP
# ============================================================
if __name__ == "__main__":
    sdr = setup_pluto()
    all_frames = []
    all_labels = []
    
    for mod in TARGET_MODS:
        print(f"\n[RECORDING] Collecting {mod} data...")
        frames_collected = 0
        
        # Clear any existing buffer before starting new modulation
        try:
            sdr.tx_destroy_buffer()
        except:
            pass

        # Generate and filter
        raw_syms = get_symbols(mod, num_symbols=4096)
        tx_sig = apply_tx_filter(raw_syms)
        
        # Start TX (scaled for Pluto DAC)
        sdr.tx(tx_sig * 10000) 
        time.sleep(0.5) 
        
        while frames_collected < FRAMES_PER_MOD:
            rx_sig = sdr.rx()
            
            # DSP
            rx_sig = rx_sig - np.mean(rx_sig)
            psd = np.abs(np.fft.fftshift(np.fft.fft(rx_sig)))
            freqs = np.fft.fftshift(np.fft.fftfreq(len(rx_sig), d=1/FS))
            peak_freq = freqs[np.argmax(psd)]
            t = np.arange(len(rx_sig)) / FS
            rx_sig = rx_sig * np.exp(-1j * 2 * np.pi * peak_freq * t)
            rx_sig = rx_sig / (np.sqrt(np.mean(np.abs(rx_sig)**2)) + 1e-9)
            
            num_slices = len(rx_sig) // FRAME_LEN
            for i in range(num_slices):
                if frames_collected >= FRAMES_PER_MOD: break
                frame = rx_sig[i*FRAME_LEN : (i+1)*FRAME_LEN]
                
                if np.max(np.abs(frame)) > 0.1:
                    I, Q = np.real(frame), np.imag(frame)
                    all_frames.append(np.stack([I, Q], axis=1))
                    all_labels.append(mod)
                    frames_collected += 1
                    
        print(f"  -> {mod} complete: {frames_collected} frames.")

    # Final Cleanup
    sdr.tx_destroy_buffer()
    
    print("\n[SAVING] Writing to disk...")
    np.save("pluto_X_train.npy", np.array(all_frames).astype(np.float32))
    np.save("pluto_Y_train.npy", np.array(all_labels))
    print("✅ SUCCESS: Dataset generated without buffer conflicts!")