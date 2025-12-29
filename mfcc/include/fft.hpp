#pragma once
#include <vector>
#include <complex>
#include <cmath>

typedef std::complex <float> cf;
const float PI = 3.1415927f;


std::vector<float> power_spectrum(const std::vector<cf> &X);
// Power spectrum for real-valued signals: returns only non-negative frequencies.
// Output size is N/2 + 1 where N = X.size().
std::vector<float> power_spectrum_rfft(const std::vector<cf> &X);
std::vector<cf> to_complex(const std::vector<float> &real_vector);
void fft(std::vector<cf> &a);