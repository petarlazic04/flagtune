import numpy as np
from scipy.fftpack import dct
import time

# ===============================
# CONFIG
# ===============================
FS = 16000
FRAME_SIZE = 512
N_MELS = 26
N_MFCC = 13
SEED = int(time.time())

INPUT_PATH = "data/input.txt"
OUTPUT_PATH = "data/out_py.txt"

# ===============================
# GENERATE + READ INPUT
# ===============================
np.random.seed(SEED)
x = 0.7*np.random.randn(FRAME_SIZE)
np.savetxt(INPUT_PATH, x)
x = np.loadtxt(INPUT_PATH)

# ===============================
# MODULES
# ===============================

def windowing(x):
    return x * np.hanning(len(x))

def power_spectrum(x):
    X = np.fft.rfft(x)
    return (np.abs(X) ** 2) / len(x)

def mel_filterbank(pspec, fs, n_mels):
    n_fft = (len(pspec) - 1) * 2
    # placeholder – pravi se poseban modul
    return pspec[:n_mels]

def log_energy(x, eps=1e-12):
    return np.log(x + eps)

def dct_mfcc(x, n_mfcc):
    return dct(x, type=2, norm='ortho')[:n_mfcc]

# ===============================
# PIPELINE (ENABLE ONE)
# ===============================

y = windowing(x)
# y = power_spectrum(windowing(x))
# y = mel_filterbank(power_spectrum(windowing(x)), FS, N_MELS)
# y = log_energy(mel_filterbank(power_spectrum(windowing(x)), FS, N_MELS))
# y = dct_mfcc(log_energy(mel_filterbank(power_spectrum(windowing(x)), FS, N_MELS)), N_MFCC)

# ===============================
# WRITE OUTPUT
# ===============================
np.savetxt(OUTPUT_PATH, y)
