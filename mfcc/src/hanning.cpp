#include "../include/hanning.hpp"



void apply_hann_window(std::vector<float>& frame){
    int N = frame.size();

    for (int n = 0; n < N; ++n) {
        frame[n] *= hann(n, N);  
    }
}