"""
   use second-order IIR filter to remove 10Hz component of signal
"""

import numpy as np
import matplotlib.pyplot as plt


def genData(fs, f1, f2, f3):
    t = np.arange(0, 1, 1/fs)

    x1 = np.cos(2*np.pi*f1*t)
    x2 = np.cos(2*np.pi*f2*t)
    x3 = np.cos(2*np.pi*f3*t)
    x = x1 + x2 + x3

    return x


def applyNotch(x, fs, f1, f2, f3) :

    f = f2           
    w = 2*np.pi*f/fs  
    
    N = x.size
    i = np.arange(N)
    
    # create the past values for x and y
    # this adds 2 additional values at the beginning of the array that must
    # be compensated for when plotting

    # filter in theory produces values forever; will stop
    #   after 100 extra values
    extension = 100
    x = np.append([0, 0], x)
    y = np.zeros(N + 2 + extension)
    
    n = 2
    while n < N+2+extension :
        if n < N+2 :
            y[n] = 1.8744*np.cos(w)*y[n-1] - 0.8783*y[n-2] + x[n] - 2*np.cos(w)*x[n-1] + x[n-2]
        else :
            y[n] = 1.8744*np.cos(w)*y[n-1] - 0.8783*y[n-2]
    
        n += 1
    
    plt.plot(x[2:])
    plt.xlim([-25, 625])  # makes comparison to other plots easier
    plt.title("original signal: %dHz + %dHz + %dHz" % (f1, f2, f3))
    
    plt.figure()
    plt.plot(y[2:])
    plt.ylim([-2.25, 2.25])  # makes comparison to next plot easier
    plt.title("filtered signal: %dHz removed" % f2)
    
    # produce clean signal
    t = np.arange(0, 1, 1/fs)

    x1 = np.cos(2*np.pi*f1*t)
    x3 = np.cos(2*np.pi*f3*t)
    x = x1 + x3
    
    plt.figure()
    plt.plot(x1+x3)
    plt.title("%dHz + %dHz signal" % (f1, f3))
    plt.xlim([-25, 625])  # makes comparison to next plot easier
    plt.tight_layout()
    plt.show()

def main():
    f1 = 5
    f2 = 17
    f3 = 43
    fs = 500

    signal = genData(fs, f1, f2, f3)
    applyNotch(signal, fs, f1, f2, f3)


if __name__ == "__main__":
    main()

