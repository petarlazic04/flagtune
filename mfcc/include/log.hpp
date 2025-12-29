#pragma once

#include <vector>

// Element-wise natural-log compression of Mel energies.
// Equivalent to NumPy: np.log(x + eps)
//
// - eps must be > 0 to avoid log(0)
// - values < 0 are clamped to 0 before applying log
void log_energy_inplace(std::vector<float>& mel_energies, float eps = 1e-12f);

std::vector<float> log_energy(const std::vector<float>& mel_energies, float eps = 1e-12f);

// Convenience: Mel energies (from mel.cpp) -> log-mel energies.
// This is the direct “mel output becomes log input” step.
std::vector<float> log_mel_energies_from_power_spectrum(
	const std::vector<float>& power_spectrum,
	const std::vector<std::vector<float>>& filterbank,
	float eps = 1e-12f
);

// Convenience: build Mel filterbank + apply + log.
// NOTE: For real-time, prefer building the filterbank once and calling
// log_mel_energies_from_power_spectrum(...) per frame.
std::vector<float> log_mel_energies(
	const std::vector<float>& power_spectrum,
	int sample_rate,
	int n_fft,
	int n_mels,
	float fmin_hz,
	float fmax_hz,
	float eps = 1e-12f
);
