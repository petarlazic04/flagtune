#include "dct.hpp"
#include "log.hpp"
#include "mel.hpp"
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
