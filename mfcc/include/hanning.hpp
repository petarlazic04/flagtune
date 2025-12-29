#pragma once

#include <vector>
#include <math.h>

inline float hann(int n, int N)
{
    if (N <= 1) return 1.0f;

    return 0.5f * (1.0f - cos(
        2.0f * static_cast<float>(M_PI) *
        static_cast<float>(n) /
        static_cast<float>(N - 1)
    ));
}


void apply_hann_window(std::vector<float>& frame);
