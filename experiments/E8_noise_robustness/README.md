# Experiment E8: Environmental Noise Robustness

## Objective
Evaluate the degradation characteristics of compressed streaming speech models under real-world acoustic noise conditions across varying Signal-to-Noise Ratios (SNRs).

## Hypothesis
Aggressively quantized (INT4) and highly pruned (> 60%) models will exhibit higher sensitivity to acoustic noise (steeper WER degradation curves at low SNRs $\le 5\text{ dB}$) than FP32/INT8 models due to reduced representational redundancy.

## Variables
* **Independent Variables**: 
  * Noise type $\in \{\text{traffic}, \text{office}, \text{crowd/babble}, \text{fan}, \text{wind}, \text{music}\}$.
  * Signal-to-Noise Ratio (SNR) $\in \{\text{clean}, 20\text{ dB}, 10\text{ dB}, 5\text{ dB}, 0\text{ dB}\}$.
* **Dependent Variables**: WER, CER degradation ($\Delta \text{WER}$), VAD false rejection / false trigger rate.
* **Control Variables**: Audio test split, compression level, chunk size (400 ms).

## Dataset
* Common Voice Swahili test set mixed with environmental noise corpora at calibrated SNR levels.

## Model
* Baseline FP32, Quantized INT8, and Pruned+INT8 streaming models.

## Procedure
1. Synthesize noisy evaluation sets at target SNRs (0, 5, 10, 20 dB, clean).
2. Run streaming speech recognition across all noise conditions.
3. Compute WER curves as a function of SNR.

## Metrics
* WER vs. SNR curves, CER, Noise Sensitivity Index ($\Delta \text{WER} / \Delta \text{SNR}$).

## Expected Output Files
* `results/raw/E8_noise_traffic_snr10.json`
* `results/raw/E8_noise_office_snr10.json`
* `results/tables/E8_noise_robustness_summary.csv`
