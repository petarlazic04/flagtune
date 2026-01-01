#include "../include/log.hpp"
#include "../include/mel.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iomanip>
#include <iostream>

void log_energy_inplace(std::vector<float>& mel_energies, float eps)
{
	if (!(eps > 0.0f))
		throw std::invalid_argument("log_energy_inplace: eps must be > 0");

	for (float& x : mel_energies)
	{
		const float clamped = std::max(0.0f, x);
		x = static_cast<float>(std::log(static_cast<double>(clamped + eps)));
	}
}

std::vector<float> log_energy(const std::vector<float>& mel_energies, float eps)
{
	std::vector<float> out = mel_energies;
	log_energy_inplace(out, eps);
	return out;
}

std::vector<float> log_mel_energies_from_power_spectrum(
	const std::vector<float>& power_spectrum,
	const std::vector<std::vector<float>>& filterbank,
	float eps
)
{
	std::vector<float> mel = mel_energies_from_power_spectrum(power_spectrum, filterbank);
	log_energy_inplace(mel, eps);
	return mel;
}

std::vector<float> log_mel_energies(
	const std::vector<float>& power_spectrum,
	int sample_rate,
	int n_fft,
	int n_mels,
	float fmin_hz,
	float fmax_hz,
	float eps
)
{
	std::vector<float> mel = mel_energies(power_spectrum, sample_rate, n_fft, n_mels, fmin_hz, fmax_hz);
	log_energy_inplace(mel, eps);
	return mel;
}

/*

int main()
{
	const int sample_rate = 16000;
	const int n_fft = 512;
	const int n_bins = n_fft / 2 + 1;
	const int n_mels = 12;

	const float fmin = 0.0f;
	const float fmax = sample_rate / 2.0f;

	// Fake POWER spectrum with a few peaks.
	std::vector<float> power(n_bins, 0.0f);
	power[16]  = 0.25f;   // 500 Hz
	power[40]  = 1.00f;   // 1250 Hz
	power[72]  = 0.64f;   // 2250 Hz
	power[104] = 0.09f;   // 3250 Hz
	power[160] = 0.36f;   // 5000 Hz
	power[224] = 0.81f;   // 7000 Hz

	std::cout << "sample_rate=" << sample_rate
			  << " n_fft=" << n_fft
			  << " n_bins=" << n_bins
			  << " n_mels=" << n_mels
			  << " fmin=" << fmin
			  << " fmax=" << fmax << "\n\n";

	auto fb = build_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, true);

	auto mel = mel_energies_from_power_spectrum(power, fb);
	auto logmel = log_energy(mel, 1e-12f);

	std::cout << "Mel energies -> log energies:\n";
	for (int m = 0; m < n_mels; ++m)
	{
		std::cout << "  m=" << std::setw(2) << m
				  << "  mel=" << std::fixed << std::setprecision(8) << mel[m]
				  << "  logmel=" << std::fixed << std::setprecision(8) << logmel[m]
				  << "\n";
	}

	return 0;
}

*/