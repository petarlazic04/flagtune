import numpy as np
import librosa

# ===============================
# PARAMETRI
# ===============================
FS = 16000
FRAME_SIZE = 512
N_MELS = 26
N_MFCC = 13

# ===============================
# GENERISI SIGNAL
# ===============================
t = np.arange(FRAME_SIZE) / FS
x = 0.7 * np.sin(2 * np.pi * 1000 * t)

print(x)

# ===============================
# MFCC – SAMO JEDAN FREJM
# ===============================
mfcc = librosa.feature.mfcc(
    y=x,
    sr=FS,
    n_mfcc=N_MFCC,
    n_fft=FRAME_SIZE,
    hop_length=FRAME_SIZE,   # nema hopovanja
    n_mels=N_MELS,
    window="hann",
    center=False
)

# mfcc shape: (13, 1)
mfcc = mfcc[:, 0]

# ===============================
# ISPIS
# ===============================
print("MFCC (1 frejm, 13 koeficijenata):")
for i, v in enumerate(mfcc):
    print(f"c{i:02d} = {v:.6f}")
