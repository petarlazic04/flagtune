#pragma once

#include <vector>

// Builds Mel filterbank weights
std::vector<std::vector<float>> build_mel_filterbank(
    int sample_rate,
    int n_fft,
    int n_mels,
    float fmin_hz,
    float fmax_hz,
    bool normalize_by_sum
);

// Applies filterbank to a POWER spectrum to get Mel energies
std::vector<float> mel_energies_from_power_spectrum(
    const std::vector<float>& power_spectrum,
    const std::vector<std::vector<float>>& filterbank
);

// Convenience function: builds filterbank and applies it
std::vector<float> mel_energies(
    const std::vector<float>& power_spectrum,
    int sample_rate,
    int n_fft,
    int n_mels,
    float fmin_hz,
    float fmax_hz
);