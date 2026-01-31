"""
   shelving filter using IIR

   based on book DSP Filters (Electronics Cookbook Series) by Lane et al.
   chapter 11 -- change gain of low frequencies
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def fft(x):
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N == 1:
        return x

    if N % 2 == 0:
        X_even = fft(x[::2])
        X_odd = fft(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        first = X_even + factor[:N // 2] * X_odd
        second = X_even + factor[N // 2:] * X_odd
        return np.concatenate([first, second])
    else:
        n = np.arange(N)
        k = n.reshape((N, 1))
        W = np.exp(-2j * np.pi * k * n / N)
        return W.dot(x)


def comparePlots(x, fs, N, y):
    xfft = abs(np.fft.fft(x))
    yfft = abs(np.fft.fft(y))


    if max(xfft) > max(yfft):
        maxAmp = max(xfft) + 100
    else:
        maxAmp = max(yfft) + 100

    frequencies = np.arange(N) * fs / N
    plt.subplot(1, 2, 1)
    plt.plot(frequencies[0:N // 4], xfft[0:N // 4])
    plt.title('original signal')
    plt.xlabel('Hz')
    plt.ylim([0, maxAmp])

    plt.subplot(1, 2, 2)
    plt.plot(frequencies[0:N // 4], yfft[0:N // 4])
    plt.title('filtered signal')
    plt.xlabel('Hz')
    plt.ylim([0, maxAmp])
    plt.show()


def applyShelvingFilter(inName, outName, g, fc):
    x, fs = sf.read(inName)
    N = x.size

    w = 2 * np.pi * fc / fs
    mu = 10 ** (g / 20)
    gamma = (1 - 4 / (1 + mu) * np.tan(w / 2)) / (1 + 4 / (1 + mu) * np.tan(w / 2))
    alpha = (1 - gamma) / 2

    u = np.zeros(N)
    y = np.zeros(N)

    u[0] = alpha * x[0]
    y[0] = x[0] + (mu - 1) * u[0]

    for n in np.arange(1, N):
        u[n] = alpha * (x[n] + x[n - 1]) + gamma * u[n - 1]
        y[n] = x[n] + (mu - 1) * u[n]

    comparePlots(x, fs, N, y)
    sf.write(outName, y, fs)


def main():
    inName = "song-Hold on a Sec.wav"
    gain = -20
    cutoff = 300
    outName = "shelvingOutput.wav"

    applyShelvingFilter(inName, outName, gain, cutoff)


if __name__ == "__main__":
    main()
