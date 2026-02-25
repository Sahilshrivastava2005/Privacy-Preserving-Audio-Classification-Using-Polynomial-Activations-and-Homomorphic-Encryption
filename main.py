from input_representation import (
    load_audio,
    compute_stft,
    compute_spectrogram,
    compute_mel_spectrogram,
    compute_mfcc    
)

import matplotlib.pyplot as plt
import librosa.display

AUDIO_PATH = "data/sample.wav"


def visualize(feature, title, y_axis="linear"):
    plt.figure(figsize=(8, 4))
    librosa.display.specshow(feature, x_axis="time", y_axis=y_axis)
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.show()


def main():

    print("Loading raw audio...")
    y, sr = load_audio(AUDIO_PATH)
    print("Raw signal shape:", y.shape)

    print("\nComputing STFT (Eq. 1)...")
    stft = compute_stft(y)
    print("STFT shape:", stft.shape)

    print("\nComputing Spectrogram (Eq. 2)...")
    spectrogram = compute_spectrogram(stft)
    print("Spectrogram shape:", spectrogram.shape)

    print("\nComputing Mel Spectrogram (Eq. 3)...")
    log_mel = compute_mel_spectrogram(spectrogram, sr)
    print("Mel Spectrogram shape:", log_mel.shape)

    print("\nComputing MFCC (Eq. 4)...")
    mfcc = compute_mfcc(log_mel)
    print("MFCC shape:", mfcc.shape)

    # Visualizations
    visualize(log_mel, "Log-Mel Spectrogram", y_axis="mel")
    visualize(mfcc, "MFCC")


if __name__ == "__main__":
    main()
