import numpy as np
import librosa
from scipy.fftpack import dct


# --------------------------------------------------
# 1️⃣ Load Raw Audio (1D time series)
# --------------------------------------------------
def load_audio(path, sr=16000):
    y, sr = librosa.load(path, sr=sr)
    return y, sr


# --------------------------------------------------
# 2️⃣ STFT  (Equation 1)
# S(f,t) = ∫ s(τ) w(τ−t) e^{-j2πfτ} dτ
# --------------------------------------------------
def compute_stft(y, n_fft=1024, hop_length=512):
    stft = librosa.stft(
        y,
        n_fft=n_fft,
        hop_length=hop_length,
        window="hann"
    )
    return stft


# --------------------------------------------------
# 3️⃣ Spectrogram (Equation 2)
# |S(f,t)|^2
# --------------------------------------------------
def compute_spectrogram(stft_matrix):
    spectrogram = np.abs(stft_matrix) ** 2
    return spectrogram


# --------------------------------------------------
# 4️⃣ Mel Spectrogram (Equation 3)
# α = 2595 log10(1 + f/700)
# --------------------------------------------------
def compute_mel_spectrogram(spectrogram, sr, n_mels=64):
    
    # Create Mel filter bank
    mel_filter = librosa.filters.mel(
        sr=sr,
        n_fft=(spectrogram.shape[0] - 1) * 2,
        n_mels=n_mels
    )

    # Apply Mel filter bank
    mel_spectrogram = np.dot(mel_filter, spectrogram)

    # Convert to log scale
    log_mel = np.log(mel_spectrogram + 1e-9)

    return log_mel


# --------------------------------------------------
# 5️⃣ MFCC (Equation 4)
# MFCC_c = Σ log(S(α,t)) cos( πc(2α+1)/2A )
# --------------------------------------------------
def compute_mfcc(log_mel_spectrogram, n_mfcc=13):

    # Apply DCT along Mel-frequency axis
    mfcc = dct(
        log_mel_spectrogram,
        type=2,
        axis=0,
        norm='ortho'
    )

    return mfcc[:n_mfcc]
