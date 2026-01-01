#include <fft.hpp>


std::vector<cf> to_complex(const std::vector<float> &real_vector) {
    std::vector<cf> complex_vector;
    complex_vector.reserve(real_vector.size()); // optimizacija

    for (float val : real_vector)
        complex_vector.push_back(cf(val, 0.0f));

    return complex_vector;
}

void fft(std::vector<cf> &a) {
    int N = a.size();
    if (N <= 1) return;

    // dijeli na parne i neparne
    std::vector<cf> even(N/2), odd(N/2);
    for (int i = 0; i < N/2; i++) {
        even[i] = a[i*2];
        odd[i]  = a[i*2 + 1];
    }

    fft(even);
    fft(odd);

    // kombinuje rezultate
    for (int k = 0; k < N/2; k++) {
        cf t = std::polar(1.0f, -2 * PI * k / N) * odd[k]; // twiddle factor
        a[k]       = even[k] + t;
        a[k + N/2] = even[k] - t;
    }
}

//Radi za kompleksne vrijednosti
std::vector<float> power_spectrum(const std::vector<cf> &X) {
    int N = X.size();
    std::vector<float> P(N);
    for (int k = 0; k < N; k++)
        P[k] = norm(X[k]) / N;  // |X[k]|^2 normalizirano
    return P;
}

//Radi za realne vrijednosti spektra sto je optimizovanije za nas pristup jer nam kompleksna polovina spektra ne radi nista
std::vector<float> power_spectrum_rfft(const std::vector<cf> &X) {
    const int N = static_cast<int>(X.size());
    if (N <= 0) return {};

    const int n_bins = N / 2 + 1;
    std::vector<float> P(static_cast<std::size_t>(n_bins));
    for (int k = 0; k < n_bins; ++k)
        P[static_cast<std::size_t>(k)] = norm(X[static_cast<std::size_t>(k)]) / N;
    return P;
}

