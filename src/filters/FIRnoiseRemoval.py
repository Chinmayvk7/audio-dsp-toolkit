"""
    FIR filter design
      from http://www.labbookpages.co.uk/audio/firWindowing.html
    remove noise using low-pass filter
"""

from scipy.signal import freqz
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

######################################
def lowpass(L, fc, fs) :
    M = L - 1
    ft = fc/fs
    h = np.zeros(L)

    n = 0
    while n <= M :
        if n == M/2 :
            h[n] = 2*ft
        else :
            h[n] = np.sin(2*np.pi*ft*(n - M/2)) / (np.pi*(n - M/2))
    
        n += 1

    return h

######################################
def hamming(L) :
    M = L - 1
    w = np.zeros(L)

    n = 0
    while n <= M :
        w[n] = 0.54 - 0.46*np.cos(2*np.pi*n/M)
        n += 1

    return w

######################################
def plotH(fs, coeff) :
    x, y = freqz(coeff, 1)

    #plt.figure()
    plt.plot(x, abs(y))
    plt.title("Frequency Response of LPF")

###############  main  ############### 
xbad, fs = sf.read("count12345Noise.wav")
N = xbad.size
f0 = fs/N


# filter coefficients
L = 101        # length

xfft = np.fft.fft(xbad)

fc = 6000      # this was chosen by looking at the fft

# produce lowpass filter coefficients
h = lowpass(L, fc, fs) 
plotH(fs, h)

w = hamming(L)
hWindowed = h*w  # apply window to lpf coefficients
plotH(fs, hWindowed)
plt.legend(["original", "windowed"])
plt.show() 


# show FFT before and after
xfft = np.fft.fft(xbad)

n = np.arange(0, N) * f0
plt.plot(n[0:N//2], abs(xfft[0:N//2]))
plt.title("FFT before filtering")
plt.show() 

y = np.convolve(xbad, hWindowed)

xfft = np.fft.fft(y)
plt.plot(n[0:N//2], abs(xfft[0:N//2]))
plt.title("FFT after filtering")

plt.show() 


sf.write("count12345WithoutNoise.wav", y, fs)

