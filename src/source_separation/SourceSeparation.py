"""
    given two audio files, each of which consists of a mixture
    of two audio sources, We will perform blind source separation

    We use a theory called ICA (Independent Component Analysis)
        where: X = A . S
            X: observed mixed signals
            S: unknown independent sources
            A: unknown mixing matrix

    Aim:        S = W . X
 Find unmixing matrix W such that outputs are statistically independent (non-Gaussian).           
    We do this by:

            Centering
            Whitening
            Iterative maximization of non-Gaussianity
"""


import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


def plotSignal(s, title):
    N = s.size
    plt.subplot(1, 2, 1)
    plt.plot(s)
    plt.title("time domain: %s" % title)

    plt.subplot(1, 2, 2)
    dft = np.fft.fft(s)
    plt.plot(abs(dft))
    plt.title("FFT: %s" % title)
    plt.show()


def center(X):
    return X - np.mean(X, axis=0)


def whiten(X):
    cov = np.cov(X, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    D_inv = np.diag(1.0 / np.sqrt(eigvals))
    return X @ eigvecs @ D_inv @ eigvecs.T


def ica(X, max_iter=1000, tol=1e-6):
    n_samples, n_features = X.shape
    W = np.random.rand(n_features, n_features)

    for _ in range(max_iter):
        W_old = W.copy()

        Y = X @ W.T
        g = np.tanh(Y)
        g_der = 1 - np.tanh(Y) ** 2

        W = (g.T @ X) / n_samples - np.diag(g_der.mean(axis=0)) @ W

        U, _, Vt = np.linalg.svd(W)
        W = U @ Vt

        if np.max(np.abs(np.abs(np.diag(W @ W_old.T)) - 1)) < tol:
            break

    return X @ W.T


def applyICA(x0, x1):
    X = np.c_[x0, x1]

    X = center(X)
    X = whiten(X)

    S = ica(X)
    return S


def unmixAudio(leftName, rightName):
    x0, fs = sf.read(leftName)
    x1, fs = sf.read(rightName)

    S = applyICA(x0, x1)

    s0 = 10 * S[:, 0]
    s1 = 10 * S[:, 1]

    sf.write("unmixed0.wav", s0, fs)
    sf.write("unmixed1.wav", s1, fs)

    plotSignal(x0, "original x0")
    plotSignal(x1, "original x1")
    plotSignal(s0, "unmixed source 0")
    plotSignal(s1, "unmixed source 1")


def main():
    leftName = "darinSiren0.wav"
    rightName = "darinSiren1.wav"
    unmixAudio(leftName, rightName)


if __name__ == "__main__":
    main()
