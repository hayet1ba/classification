# PlutoSDR 10-Class Classification System - Setup Complete

## What Was Updated

### 1. **train_pluto_model.py** 
✅ 10-class model setup:
- MOD_CLASSES: BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM, GMSK, OQPSK, AM-DSB-WC, FM
- Legacy class name mapping (QAM16 → 16QAM)
- Final Dense layer: 10 outputs with softmax
- Model loading with `compile=False` for compatibility

### 2. **test_transmitter.py**
✅ 10-class inference with "Autre" logic:
- CONFIDENCE_THRESHOLD = 0.7 (70% threshold)
- Enhanced signal generation for all 10 modulation types
- If confidence < 70%: Display "Autre (Unidentified)"
- If confidence ≥ 70%: Display detected class name
- UI shows ✅ HIGH CONFIDENCE or ⚠️ LOW CONFIDENCE status

### 3. **generate_pluto_dataset.py**
✅ Extended signal generation for 10 modulation types:
- Added: 64QAM, 256QAM, GMSK, OQPSK, AM-DSB-WC, FM
- Proper normalization for each constellation
- Ready for future data collection campaigns

## Current Training Status
🔄 **IN PROGRESS** - Training on existing 4-class data
- Epochs: 2/15 completed
- Model: pluto_final_classifier.h5 (being saved)
- Accuracy improving on validation set

## Next Steps After Training Completes
1. ✅ Model saved → Ready for inference
2. Run test_transmitter.py to test with PlutoSDR hardware
3. Generate new data for 6 missing classes (64QAM, 256QAM, GMSK, OQPSK, AM-DSB-WC, FM)
4. Retrain with full 10-class dataset

## Signal Generation Features
Each modulation type is now properly generated:
- **BPSK/QPSK**: Standard constellations
- **8PSK**: 8-point phase constellation
- **16QAM**: 4×4 QAM grid
- **64QAM**: 8×8 QAM grid (NEW)
- **256QAM**: 16×16 QAM grid (NEW)
- **GMSK**: Gaussian FSK with continuous phase (NEW)
- **OQPSK**: Offset QPSK with staggered I/Q (NEW)
- **AM-DSB-WC**: Double-sideband AM with carrier (NEW)
- **FM**: Frequency modulation (NEW)

## Confidence Threshold Logic
```
IF confidence ≥ 70% → AI predictions are trusted
IF confidence < 70% → Reject as "Autre" (unknown/noise)
```
This allows the system to gracefully handle:
- Noise and interference
- Unknown modulation types
- Ambiguous signals
