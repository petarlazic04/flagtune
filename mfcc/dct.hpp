#pragma once

#include <vector>

// DCT-II with orthonormal normalization.
// This is the standard transform used for MFCC when applied to log-mel energies.
//
// Input:  x (typically log-mel), size N
// Output: X, size N
std::vector<float> dct_ii_ortho(const std::vector<float>& x);

// MFCC helper:
// - Input: log_mel (output from log.cpp), size N_mels
// - Output: first n_mfcc DCT coefficients (size n_mfcc)
std::vector<float> dct_mfcc(const std::vector<float>& log_mel, int n_mfcc);
