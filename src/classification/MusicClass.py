"""
    my simple Shazam clone

    Darin Brezeale
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm
import soundfile as sf
from scipy.signal import spectrogram
import glob


def classifyMusic(metric) :
    songList = glob.glob("song-*.wav")
    songList = buildList( songList )
    
    sampleSig, fs = getSample("testSong.wav")
    
    normList = findMatch(sampleSig, songList, metric)
    normList.sort()
    if metric == "cosineSimilarity" :
        normList.reverse()  # for cosine similarity, larger is better
    
    # spectrogram of test song
    name = "testSong.wav"
    x, fs = sf.read(name)
    plt.specgram(x, Fs=fs)
    plt.title(name)

    # norms and spectrograms of other songs, 
    #   sorted by how close they match
    i = 0
    while i < len(normList) :
        value, name = normList[i]

        plt.figure()
        x, fs = sf.read(name)
        plt.specgram(x, Fs=fs)
        plt.title("%s, metric = %.3f" % (name, value))

        i += 1

    plt.show()


def buildList( songList ) :
    d = []

    for song in songList :
        x, fs = sf.read(song)
        sig = findFreq(x, fs)
        h = tuple(sig) 
        d.append([h, song])

    return d


def findFreq(x, fs) :
    #   https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    f, t, Sxx = spectrogram(x, fs=fs, nperseg=fs//2)

    signature = buildSig(f, Sxx)

    return signature


def buildSig(f, Sxx) :
    #sig = Sxx.max(axis=0) 
    sig = np.zeros( Sxx.shape[1] )

    rows = Sxx.shape[0]
    cols = Sxx.shape[1]

    c = 0
    while c < cols :
        colMax = 0
        rowIndex = 0
        r = 0
        while r < rows :
            if Sxx[r][c] > colMax :
                colMax = Sxx[r][c]
                rowIndex = r
            r += 1

        sig[c] = f[rowIndex]
        c += 1

    return sig


def getSample(name) :
    x, fs = sf.read(name)

    sig = findFreq(x, fs)
    h = tuple(sig)   # use tuple as dictionary key

    return h, fs


def findMatch(sampleSig, songList, metric) :
    normList = []
    sample = np.asarray(sampleSig)
    for song in songList :
        s = np.asarray( song[0] )
        if metric == "1-norm" :
            result = norm(s - sample, 1)
        elif metric == "2-norm" :
            result = norm(s - sample, 2)
        elif metric == "cosineSimilarity" :
            result = (np.dot(s, sample))/(norm(s)*norm(sample))
  
        normList.append([result, song[1]])

    return normList
 

def main():
    metric = "cosineSimilarity"
    classifyMusic(metric)


if __name__ == "__main__":
    main()
