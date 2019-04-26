# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np
import numpy, scipy, IPython.display as ipd, matplotlib.pyplot as plt

def plot_audio(y, sr, filename):
    N = 1024
    f_max = 17000

    #Computar o espectro de potência do sinal de áudio
    stft = np.abs(librosa.stft(y, n_fft=N, window='hann'))**2
    mel_stft = librosa.feature.melspectrogram(S=stft, sr=sr, n_mels=128, fmax=f_max)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
    # Chromagrama por passo da classe
    librosa.display.specshow(chroma_cqt, y_axis='chroma', x_axis='time')
    # Espectrograma por Hz
    # librosa.display.specshow(chroma_cqt, y_axis='cqt_hz', x_axis='time')
    # Espectrograma por Nota
    # librosa.display.specshow(chroma_cqt, y_axis='cqt_note')

    plt.title("chroma_cqt")
    plt.show()
    


if __name__ == '__main__':
    from sys import argv
    
    filename = argv[1]
    x, sr = librosa.load(filename)
    plot_audio(x, sr, filename)

    print("Concluido!")