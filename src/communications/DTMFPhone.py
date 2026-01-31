"""
    create filter bank for decoding dual tone multifrequency (DTMF) signals

"""

from scipy.signal import freqz, spectrogram
import numpy as np
import matplotlib.pyplot as plt



def plotBandpass(frequencies, L, fs) :
    plt.figure()

    # plot frequency responses
    for f in frequencies :
        h = filterCoefficients(f, L, fs)
        x, y = freqz(h, 1)

        fLabels = fs * x / (2 * np.pi)
        plt.plot(fLabels, abs(y))
    
    plt.title( "Frequency Responses of Bandpass Filters")
    plt.xlabel("Hertz")
    plt.savefig("bandpass.eps", bbox_inches="tight")



def filterCoefficients(fb, L, fs) :
    n = np.arange(0, L)
    h = 2/L*np.cos(2*np.pi*fb*n/fs) 

    return h


def processTones(filename, L, fs, samplesPerTone) :
    signal = np.genfromtxt(filename, delimiter=',')
    
    freqRow = [ 697,  770,  852,  941]
    freqCol = [1209, 1336, 1477]
    
    numbers = [['1', '2', '3'],
               ['4', '5', '6'],
               ['7', '8', '9'],
               ['*', '0', '#']]
    
    phoneNumber = ""
    
    plotBandpass(np.append(freqRow, freqCol), L, fs)
    # use default window size
    plotSpec(signal, fs)
    plt.show()

    
    q = signal.size//samplesPerTone  # use integer division
    
    for i in range(0, q) :
        x = signal[i*samplesPerTone : (i+1)*samplesPerTone]
        maxValue = 0
        maxRow = 0
        for r in range(0, 4) :
            fb = freqRow[r]
            h = filterCoefficients(fb, L, fs)
            y = np.convolve(x, h)
            y2mean = np.mean(y**2)
            if y2mean > maxValue :
                maxValue = y2mean
                maxRow = r
    
        maxValue = 0
        maxCol = 0
        for c in range(0, 3) :
            fb = freqCol[c]
            h = filterCoefficients(fb, L, fs)
            y = np.convolve(x, h)
            y2mean = np.mean(y**2)
            if y2mean > maxValue :
                maxValue = y2mean
                maxCol = c
    
        s = numbers[maxRow][maxCol]
        phoneNumber = phoneNumber + s
    
    return phoneNumber


def plotSpec(x, fs) :
    f, t, Sxx = spectrogram(x, fs)

    plt.figure()
    plt.pcolormesh(t, f, Sxx)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.savefig("spectrogram.eps", bbox_inches="tight")

def main():
    filename = "tones.csv"
    L = 64                      #  filter length
    fs = 8000                   #  sampling rate
    samplesPerTone = 4000       #  4000 samples per tone,

    # returns string of telephone buttons corresponding to tones
    phoneNumber = processTones(filename, L, fs, samplesPerTone)
    print(phoneNumber)


if __name__ == "__main__":
    main()

