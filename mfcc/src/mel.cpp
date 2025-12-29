#include "../include/mel.hpp"
#include "../include/fft.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <iostream>
#include <iomanip>

double hz_to_mel(double hz) {  
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

double mel_to_hz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

int clamp_int(int v, int lo, int hi) {
    return std::min(hi, std::max(lo, v));
}

// Builds Mel filterbank weights
std::vector<std::vector<float>> build_mel_filterbank(
    int sample_rate,
    int n_fft,
    int n_mels,
    float fmin_hz,
    float fmax_hz,
    bool normalize_by_sum
) {
    if (sample_rate <= 0 || n_fft <= 0 || n_mels <= 0) {
        throw std::invalid_argument("build_mel_filterbank: sample_rate, n_fft, n_mels must be > 0");
    }

    const int n_bins = n_fft / 2 + 1;
    const double nyquist = sample_rate / 2.0;

    double fmin = std::max(0.0, static_cast<double>(fmin_hz));
    double fmax = static_cast<double>(fmax_hz);
    if (fmax_hz <= 0.0f || fmax > nyquist) fmax = nyquist;

    if (!(fmin < fmax)) {
        throw std::invalid_argument("build_mel_filterbank: fmin must be < fmax");
    }

    const double mel_min = hz_to_mel(fmin);
    const double mel_max = hz_to_mel(fmax);

    // n_mels + 2 points define n_mels triangles
    std::vector<double> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const double t = static_cast<double>(i) / (n_mels + 1);
        mel_points[i] = mel_min + t * (mel_max - mel_min);
    }

    // Convert to FFT-bin indices
    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        const double hz = mel_to_hz(mel_points[i]);
        int bin = static_cast<int>(std::floor((n_fft + 1) * hz / sample_rate));
        bin = clamp_int(bin, 0, n_bins - 1);
        bin_points[i] = bin;
    }

    std::vector<std::vector<float>> fb(n_mels, std::vector<float>(n_bins, 0.0f));

    for (int m = 0; m < n_mels; ++m) {
        int left = bin_points[m];
        int center = bin_points[m + 1];
        int right = bin_points[m + 2];

        // Ensure strictly increasing
        if (left == center) center = std::min(center + 1, n_bins - 1);
        if (center == right) right = std::min(right + 1, n_bins - 1);
        if (!(left < center && center < right)) {
            continue;
        }

        for (int k = left; k < center; ++k) {
            fb[m][k] = static_cast<float>((k - left) / static_cast<double>(center - left));
        }
        for (int k = center; k < right; ++k) {
            fb[m][k] = static_cast<float>((right - k) / static_cast<double>(right - center));
        }

        if (normalize_by_sum) {
            double sum = 0.0;
            for (int k = 0; k < n_bins; ++k) sum += fb[m][k];
            if (sum > 0.0) {
                for (int k = 0; k < n_bins; ++k) {
                    fb[m][k] = static_cast<float>(fb[m][k] / sum);
                }
            }
        }
    }

    return fb;
}

// Applies filterbank to a POWER spectrum to get Mel energies
std::vector<float> mel_energies_from_power_spectrum(
    const std::vector<float>& power_spectrum,
    const std::vector<std::vector<float>>& filterbank
) {
    if (filterbank.empty()) return {};

    const std::size_t n_bins = filterbank[0].size();
    if (power_spectrum.size() != n_bins) {
        throw std::invalid_argument("mel_energies_from_power_spectrum: power_spectrum size mismatch");
    }

    std::vector<float> mel(filterbank.size(), 0.0f);
    for (std::size_t m = 0; m < filterbank.size(); ++m) {
        if (filterbank[m].size() != n_bins) {
            throw std::invalid_argument("mel_energies_from_power_spectrum: filterbank row size mismatch");
        }
        double acc = 0.0;
        for (std::size_t k = 0; k < n_bins; ++k) {
            acc += static_cast<double>(filterbank[m][k]) * power_spectrum[k];
        }
        mel[m] = static_cast<float>(acc);
    }
    return mel;
}

// Convenience function: builds filterbank and applies it
std::vector<float> mel_energies(
    const std::vector<float>& power_spectrum,
    int sample_rate,
    int n_fft,
    int n_mels,
    float fmin_hz,
    float fmax_hz
) {
    std::vector<std::vector<float>> fb =
        build_mel_filterbank(sample_rate, n_fft, n_mels, fmin_hz, fmax_hz, true);
    return mel_energies_from_power_spectrum(power_spectrum, fb);
}

std::vector<float> mel_energies_from_frame(
    const std::vector<float>& frame,
    int sample_rate,
    int n_fft,
    int n_mels,
    float fmin_hz,
    float fmax_hz
) {
    if (static_cast<int>(frame.size()) != n_fft) {
        throw std::invalid_argument("mel_energies_from_frame: frame size must equal n_fft");
    }

    std::vector<cf> X = to_complex(frame);
    fft(X);
    std::vector<float> P = power_spectrum_rfft(X); // size = n_fft/2 + 1
    return mel_energies(P, sample_rate, n_fft, n_mels, fmin_hz, fmax_hz);
}

/*

int main() {
    const int sample_rate = 16000;
    const int n_fft = 512;
    const int n_mels = 26;
    const float fmin = 0.0f;
    const float fmax = sample_rate / 2.0f;

    // Simple test frame: 1 kHz sine.
    const double freq = 1000.0;
    std::vector<float> frame(n_fft);
    for (int n = 0; n < n_fft; ++n) {
        frame[n] = static_cast<float>(std::sin(2.0 * M_PI * freq * n / sample_rate));
    }

    auto mel = mel_energies_from_frame(frame, sample_rate, n_fft, n_mels, fmin, fmax);

    std::cout << "Mel energies from frame (1kHz sine):\n";
    for (int m = 0; m < n_mels; ++m) {
        std::cout << "  mel[" << std::setw(2) << m << "] = "
                  << std::fixed << std::setprecision(8) << mel[static_cast<std::size_t>(m)]
                  << "\n";
    }
    return 0;
}

*/