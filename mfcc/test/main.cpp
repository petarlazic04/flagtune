#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cmath>

#include "hanning.hpp"
#include "dft.hpp"
#include "mel.hpp"
#include "dct.hpp"

namespace fs = std::filesystem;

// ===============================
// Helpers
// ===============================

std::vector<double> read_vector(const fs::path& path)
{
    std::ifstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open input file");

    std::vector<double> v;
    double x;
    while (f >> x)
        v.push_back(x);

    return v;
}

void write_vector(const fs::path& path, const std::vector<double>& v)
{
    std::ofstream f(path);
    if (!f.is_open())
        throw std::runtime_error("Cannot open output file");

    for (double x : v)
        f << x << "\n";
}

// ===============================
// MAIN
// ===============================

int main(int argc, char** argv)
{
    try
    {
        // project_root/cpp/main.exe -> project_root
        fs::path exe_path = fs::canonical(argv[0]);
        fs::path project_root = exe_path.parent_path().parent_path();

        fs::path data_dir = project_root / "data";
        fs::path input_path = data_dir / "input.txt";
        fs::path output_path = data_dir / "out_cpp.txt";

        fs::create_directories(data_dir);

        // ===============================
        // READ INPUT
        // ===============================
        std::vector<double> x = read_vector(input_path);

        // ===============================
        // PIPELINE (ENABLE ONE)
        // ===============================

        //Brate pisi sve sa double iz nekog razloga bolje prokine

        // apply_hann_window(x);
        // x = fft_mag(x);
        // x = power_spectrum(x);
        // x = mel_filterbank(x);
        // x = log_energy(x);
        // x = dct_mfcc(x);

        // ===============================
        // WRITE OUTPUT
        // ===============================
        write_vector(output_path, x);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
