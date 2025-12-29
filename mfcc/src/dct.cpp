#include "../include/dct.hpp"
#include "../include/log.hpp"
#include "../include/mel.hpp"
#include <iomanip>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

namespace {
constexpr double kPi = 3.141592653589793238462643383279502884;
}

std::vector<float> dct_ii_ortho(const std::vector<float>& x)
{
	const int N = static_cast<int>(x.size());
	if (N == 0) return {};

	std::vector<float> out(static_cast<std::size_t>(N), 0.0f);

	const double scale0 = std::sqrt(1.0 / static_cast<double>(N));
	const double scale = std::sqrt(2.0 / static_cast<double>(N));

	// DCT-II:
	// X[k] = alpha(k) * sum_{n=0}^{N-1} x[n] * cos(pi/N * (n + 0.5) * k)
	// alpha(0)=sqrt(1/N), alpha(k)=sqrt(2/N)
	for (int k = 0; k < N; ++k)
	{
		const double a = (k == 0) ? scale0 : scale;
		double acc = 0.0;
		for (int n = 0; n < N; ++n)
		{
			const double angle = (kPi / static_cast<double>(N))
				* (static_cast<double>(n) + 0.5)
				* static_cast<double>(k);
			acc += static_cast<double>(x[static_cast<std::size_t>(n)]) * std::cos(angle);
		}
		out[static_cast<std::size_t>(k)] = static_cast<float>(a * acc);
	}

	return out;
}

std::vector<float> dct_mfcc(const std::vector<float>& log_mel, int n_mfcc)
{
	if (n_mfcc <= 0)
		throw std::invalid_argument("dct_mfcc: n_mfcc must be > 0");
	if (static_cast<int>(log_mel.size()) < n_mfcc)
		throw std::invalid_argument("dct_mfcc: n_mfcc cannot exceed input size");

	std::vector<float> full = dct_ii_ortho(log_mel);
	full.resize(static_cast<std::size_t>(n_mfcc));
	return full;
}

/*

int main()
{
	const int sample_rate = 16000;
	const int n_fft = 512;
	const int n_bins = n_fft / 2 + 1;
	const int n_mels = 26;
	const int n_mfcc = 13;

	const float fmin = 0.0f;
	const float fmax = sample_rate / 2.0f;
	const float eps = 1e-12f;

	// Fake POWER spectrum with a few peaks (same idea as log.cpp demo).
	std::vector<float> power(n_bins, 0.0f);
	power[16]  = 0.25f;   // 500 Hz
	power[40]  = 1.00f;   // 1250 Hz
	power[72]  = 0.64f;   // 2250 Hz
	power[104] = 0.09f;   // 3250 Hz
	power[160] = 0.36f;   // 5000 Hz
	power[224] = 0.81f;   // 7000 Hz

	auto fb = build_mel_filterbank(sample_rate, n_fft, n_mels, fmin, fmax, true);
	std::vector<float> logmel = log_mel_energies_from_power_spectrum(power, fb, eps);
	std::vector<float> mfcc = dct_mfcc(logmel, n_mfcc);

	std::cout << "log-mel -> MFCC demo\n";
	std::cout << "n_mels=" << n_mels << " n_mfcc=" << n_mfcc << "\n\n";

	std::cout << "Log-mel energies:\n";
	for (int i = 0; i < n_mels; ++i)
	{
		std::cout << "  logmel[" << std::setw(2) << i << "] = "
				  << std::fixed << std::setprecision(8)
				  << logmel[static_cast<std::size_t>(i)]
				  << "\n";
	}

	std::cout << "\nMFCC (DCT-II ortho, first coefficients):\n";
	for (int i = 0; i < n_mfcc; ++i)
	{
		std::cout << "  c[" << std::setw(2) << i << "] = "
				  << std::fixed << std::setprecision(8)
				  << mfcc[static_cast<std::size_t>(i)]
				  << "\n";
	}

	return 0;
}

*/