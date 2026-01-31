"""
    my simple Shazam clone
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import soundfile as sf
from scipy.signal import spectrogram
import glob


# AUDIO LOADER 
def get_base_dir():

    if "__file__" in globals():
        return os.path.dirname(os.path.abspath(__file__))
    else:
        return os.getcwd()


BASE_DIR = get_base_dir()


def getSample(name):
    file_path = os.path.join(BASE_DIR, name)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"FILE NOT FOUND: {file_path}")

    x, fs = sf.read(file_path)
    return x, fs


# MAIN CLASSIFICATION
def classifyMusic(metric):
    songList = glob.glob(os.path.join(BASE_DIR, "song-*.wav"))
    songList = buildList(songList)

    sampleSig, fs = getSample("test-Beat Thee.wav")

    normList = findMatch(sampleSig, songList, metric)
    normList.sort()
    if metric == "cosineSimilarity":
        normList.reverse()

    # spectrogram of test song
    name = "test-Beat Thee.wav"
    x, fs = getSample(name)
    plt.figure()
    plt.specgram(x, Fs=fs)
    plt.title(name)

    # matched songs
    for value, name in normList:
        plt.figure()
        x, fs = getSample(os.path.basename(name))
        plt.specgram(x, Fs=fs)
        plt.title(f"{os.path.basename(name)}, metric = {value:.3f}")

    plt.show()



def buildList(songList):
    d = []
    for song in songList:
        x, fs = sf.read(song)
        sig = findFreq(x, fs)
        d.append([tuple(sig), os.path.basename(song)])
    return d



def findFreq(x, fs):
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=fs // 2)
    return buildSig(f, Sxx)


def buildSig(f, Sxx):
    sig = np.zeros(Sxx.shape[1])
    for c in range(Sxx.shape[1]):
        sig[c] = f[np.argmax(Sxx[:, c])]
    return sig


def findMatch(sampleSig, songList, metric) :
    normList = []
    sample = np.asarray(sampleSig)

    for song in songList :
        s = np.asarray(song[0])

        # align vector lengths
        L = min(len(s), len(sample))
        s = s[:L]
        sample_cut = sample[:L]

        if metric == "1-norm" :
            result = norm(s - sample_cut, 1)
        elif metric == "2-norm" :
            result = norm(s - sample_cut, 2)
        elif metric == "cosineSimilarity" :
            result = np.dot(s, sample_cut) / (norm(s) * norm(sample_cut))

        normList.append([result, song[1]])

    return normList



def main():
    classifyMusic("cosineSimilarity")


if __name__ == "__main__":
    main()
