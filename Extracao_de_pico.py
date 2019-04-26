# -*- coding: utf-8 -*-
import matplotlib
import seaborn
import numpy, scipy, IPython.display as ipd, matplotlib.pyplot as plt
import librosa, librosa.display
plt.rcParams['figure.figsize'] = (14, 8)


# Exibe sinal CQT
def exibe_cqt(x, sr):
    cqt = librosa.cqt(x, sr=sr, n_bins=(n_octaves * bins_per_octave), bins_per_octave=bins_per_octave)
    log_cqt = librosa.amplitude_to_db(cqt)
    # librosa.display.specshow(abs(cqt), sr=sr, x_axis='time', y_axis='cqt_note')
    print(cqt.shape)
    return log_cqt
    

# Espctrograma
def expectrograma(x, log_cqt, sr, bins_per_octave):
    librosa.display.specshow(log_cqt, sr=sr, x_axis='time', y_axis='cqt_note', bins_per_octave=bins_per_octave)
    print("\n")
    plt.title('Tom de cada nota')
    plt.show()
    # plt.colorbar(format='%2.0f dB')
    

# Envelope de forca
def envelope(x, sr, hop_length):
    print("Envelope da força inical")
    onset_env = librosa.onset.onset_strength(x, sr=sr, hop_length=hop_length)
    print(onset_env)
    # plt.xlim(0, len(onset_env)) -> (0, 11210)
    

# Posicao estimada dos onsets
def posicao_onsets(x, sr, hop_length):
    print("Posicao estimada")
    onset_samples = librosa.onset.onset_detect(x, sr=sr, units='samples', hop_length=hop_length, backtrack=False, pre_max=20, post_max=20, pre_avg=100, post_avg=100, delta=0.2, wait=0)
    print(onset_samples)
    return onset_samples
    

# Concatena
def concatena(x, sr, onset_samples):
    print("Concatena onsets")
    onset_boundaries = numpy.concatenate([[0], onset_samples, [len(x)]])
    # print(onset_boundaries)
    return onset_boundaries
    

# Tempo dos onsets em seg
def tempo_onsets(sr, onset_boundaries):
    print("Tempo dos onsets")
    onset_times = librosa.samples_to_time(onset_boundaries, sr=sr)
    print(onset_times)
    return onset_times
    

# Forma da onda
def forma_onda(x, sr, onset_times):
    librosa.display.waveplot(x, sr=sr)
    plt.vlines(onset_times, -1, 1, color='r')
    plt.title('Amplitude da forma da onda')

def estimate_pitch(segment, sr, fmin=50.0, fmax=2000.0):    
    # Computa a autocorrelação do segmento de entrada.
    r = librosa.autocorrelate(segment)    
    # Defini os limites inferiores e superiores para o argmax de autocorrelação.
    i_min = sr/fmax
    i_max = sr/fmin
    r[:int(i_min)] = 0
    r[int(i_max):] = 0    
    # Encontra a localização da autocorrelação máxima. 
    i = r.argmax()
    f0 = float(sr)/i
    return f0


def generate_sine(f0, sr, n_duration):
    n = numpy.arange(n_duration)
    return 0.2*numpy.sin(2*numpy.pi*f0*n/float(sr))

def estimate_pitch_and_generate_sine(x, onset_samples, i, sr):
    n0 = onset_samples[i]
    n1 = onset_samples[i+1]
    f0 = estimate_pitch(x[n0:n1], sr)
    return generate_sine(f0, sr, n1-n0)

def concatena_sintetizado(x, sr, onset_boundaries):
    print("Tom de cada nota sintetizado")
    y = numpy.concatenate([
        estimate_pitch_and_generate_sine(x, onset_boundaries, i, sr=sr)
        for i in range(len(onset_boundaries)-1)
    ])
    # ipd.Audio(y, rate=sr)
    cqt = librosa.cqt(y, sr=sr)
    librosa.display.specshow(abs(cqt), sr=sr, x_axis='time', y_axis='cqt_note')
    plt.title('Notas sintetizadas')
    plt.show()


if __name__ == '__main__':
    from sys import argv
    
    filename = argv[1]

    x, sr = librosa.load(filename)
    hop_length = 100
    bins_per_octave = 12 * 3
    n_octaves = 7
    # ipd.Audio(x, rate=sr)
    log_cqt = exibe_cqt(x, sr)
    expectrograma(x, log_cqt, sr, bins_per_octave)
    envelope(x, sr, hop_length)
    onset_samples = posicao_onsets(x, sr, hop_length)
    onset_boundaries = concatena(x, sr, onset_samples)
    onset_times = tempo_onsets(sr, onset_boundaries)
    forma_onda(x, sr, onset_times)
    concatena_sintetizado(x, sr, onset_boundaries)


    print("\nConcluido!")