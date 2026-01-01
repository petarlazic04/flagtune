import numpy as np
import librosa
import os

# ===============================
# CONFIG
# ===============================
FS = 16000            # sample rate
FRAME_SIZE = 512
HOP_LENGTH = 256
N_MELS = 26
N_MFCC = 13

# folderi
RAW_AUDIO_DIR = "data/raw_audio/"
MFCC_DIR = "data/mfcc"
os.makedirs(MFCC_DIR, exist_ok=True)

# lista audio fajlova i imena klasa
audio_files = {
    "srb": "gusle.wav",
    "ind": "indija.wav",
    "mex": "meksiko.wav",
    "jam": "jamajka.wav",
    "rom": "rumunija.wav"
}

# ===============================
# GENERISANJE MFCC MATRICA
# ===============================
for class_name, filename in audio_files.items():
    path = os.path.join(RAW_AUDIO_DIR, filename)
    
    # učitaj audio
    y, sr = librosa.load(path, sr=FS)
    
    # izračunaj MFCC
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=FRAME_SIZE,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS
    ).T  # transpose da dobijemo shape (frames, 13)
    
    # sačuvaj kao .npy
    out_path = os.path.join(MFCC_DIR, f"{class_name}.npy")
    np.save(out_path, mfcc)
    print(f"Saved {out_path}, shape = {mfcc.shape}")
