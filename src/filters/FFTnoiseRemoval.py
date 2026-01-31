"""
use FFT to remove noise from audio file

"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import math


def next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    return 1 << (n - 1).bit_length()


def bit_reverse_indices(n: int) -> np.ndarray:
    """Return bit-reversed indices for length n (n must be power of two)."""
    bits = int(math.log2(n))
    indices = np.arange(n)
    rev = np.zeros(n, dtype=int)
    for i in range(n):
        b = format(i, '0{}b'.format(bits))
        rev[i] = int(b[::-1], 2)
    return rev


def fft_iterative(x: np.ndarray) -> np.ndarray:
    """
    Radix-2 iterative Cooley-Tukey FFT.
    Input x: 1D complex numpy array, length must be a power of two.
    Returns X: complex numpy array of same length.
    """
    n = x.shape[0]
    if n == 1:
        return x.copy()
    if (n & (n - 1)) != 0:
        raise ValueError("fft_iterative requires length to be a power of two")

    # Bit-reverse copy
    rev = bit_reverse_indices(n)
    a = x[rev].astype(np.complex128, copy=True)

    # Iterative Cooley-Tukey
    m = 1
    while m < n:
        m2 = 2 * m
        # w_m = exp(-2j*pi/(m2))
        theta = -2.0 * math.pi / m2
        w_m_real = math.cos(theta)
        w_m_imag = math.sin(theta)
        for k in range(0, n, m2):
            wr, wi = 1.0, 0.0  # current twiddle factor (real, imag)
            for j in range(m):
                t_real = wr * a[k + j + m].real - wi * a[k + j + m].imag
                t_imag = wr * a[k + j + m].imag + wi * a[k + j + m].real
                u_real = a[k + j].real
                u_imag = a[k + j].imag
                a[k + j] = complex(u_real + t_real, u_imag + t_imag)
                a[k + j + m] = complex(u_real - t_real, u_imag - t_imag)
                # multiply wr+ i*wi by w_m (complex multiply)
                tmp_wr = wr * w_m_real - wi * w_m_imag
                tmp_wi = wr * w_m_imag + wi * w_m_real
                wr, wi = tmp_wr, tmp_wi
        m = m2
    return a


def ifft_iterative(X: np.ndarray) -> np.ndarray:
    """
    Inverse FFT using conjugation and fft_iterative:
    ifft(x) = conj( fft( conj(x) ) ) / n
    """
    n = X.shape[0]
    # Conjugate input, FFT, conjugate result, divide by n
    conj_in = np.conjugate(X)
    y = fft_iterative(conj_in)
    y = np.conjugate(y) / n
    return y


def displaySignal(signal, fs, N, f0, s):
    """
    Plot time-domain signal and its magnitude spectrum (frequency axis).
    - signal: 1D numpy array (length N)
    - fs: sampling frequency
    - N: length used for FFT (power-of-two, possibly >= len(signal))
    - f0: nominal frequency resolution (fs / N)
    - s: title suffix
    """
    plt.figure(figsize=(10, 3))
    plt.plot(signal)
    plt.title("%s — time domain (fs = %d Hz)" % (s, fs))
    plt.xlabel("sample index")
    plt.ylabel("amplitude")

    # Compute FFT for plotting (pad/truncate as needed)
    sig_for_fft = np.zeros(N, dtype=np.complex128)
    sig_for_fft[: signal.size] = signal
    X = fft_iterative(sig_for_fft)

    # Use frequency axis (0 .. fs)
    freqs = np.linspace(0.0, fs, N, endpoint=False)

    # Plot magnitude spectrum (only first half is informative for real signals)
    half = N // 2
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[:half], np.abs(X[:half]) )
    plt.title("Magnitude spectrum of %s (0 to Nyquist %.1f Hz)" % (s, fs / 2.0))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    # Set x ticks (at most 10 ticks)
    num_ticks = 10
    tick_locs = np.linspace(0, freqs[half - 1], min(num_ticks, half), dtype=int)
    plt.xticks(tick_locs, [f"{int(t)}" for t in freqs[tick_locs]])
    plt.tight_layout()
    plt.show()


def removeNoise(signal, fs, N, f0):
    """
    Remove a band of frequencies around the center (as per original logic).
    This function uses the custom fft/ifft.

    - signal: 1D numpy array (original length)
    - fs: sampling frequency
    - N: FFT length used (power-of-two)
    - f0: fs / N (frequency resolution)
    """
    # Convert to complex and pad to N
    x_padded = np.zeros(N, dtype=np.complex128)
    x_padded[: signal.size] = signal

    # The midpoint index and a symmetric window are used (like original)
    mid = N // 2
    # The offset from original code — guard it to be <= mid-1
    offset = 73000
    if offset >= mid:
    # If offset too large, clamp to half-spectrum minus 1
        offset = mid - 1
    lowerBound = max(0, mid - offset)
    upperBound = min(N, mid + offset + 1)
    print("FFT length (N) = %d, midpoint = %d, lower = %d, upper = %d" % (N, mid, lowerBound, upperBound))

    X = fft_iterative(x_padded)

    # Zero out a symmetric band around midpoint (this removes those frequency bins)
    X[lowerBound:upperBound] = 0

    # Inverse FFT
    cleaned_full = ifft_iterative(X)

    # Return real part trimmed to original signal length
    cleaned = cleaned_full.real[: signal.size]
    return cleaned



if __name__ == "__main__":
    filename = "count12345Noise.wav"
    try:
        signal, fs = sf.read(filename)
    except Exception as e:
        raise SystemExit(f"Error reading '{filename}': {e}")

    # If stereo, convert to mono by averaging channels (preserve shape as 1D)
    if signal.ndim == 2:
        print("Input is stereo — converting to mono by averaging channels.")
        signal = signal.mean(axis=1)

    N_orig = signal.size
    # Choose FFT length as next power-of-two >= original length (radix-2 requirement)
    N_fft = next_power_of_two(N_orig)
    f0 = fs / N_fft

    print(f"Original length = {N_orig}, FFT length used = {N_fft}, frequency resolution f0 = {f0:.6f} Hz")

    # before removing noise
    displaySignal(signal, fs, N_fft, f0, "signal with noise")

    cleaned = removeNoise(signal, fs, N_fft, f0)
    # Ensure within valid audio range (-1..1 or int16). We'll keep float32 in [-1,1] if original was float.
    # Cast to same dtype as original if it's float; if original was int, scale might be necessary.
    cleaned_to_write = cleaned.astype(np.float32)

    out_filename = "count12345WithoutNoise.wav"
    sf.write(out_filename, cleaned_to_write, fs)
    print(f"Wrote cleaned audio to '{out_filename}' (trimmed to original length).")

    # after removing noise
    displaySignal(cleaned, fs, N_fft, f0, "after noise removed")

    # look at the original audio before noise was added (if you have a clean file)
    orig_filename = "count12345.wav"
    try:
        orig_signal, orig_fs = sf.read(orig_filename)
        if orig_signal.ndim == 2:
            orig_signal = orig_signal.mean(axis=1)
        # Use same N_fft and f0 for fair comparison (pad/truncate inside displaySignal)
        displaySignal(orig_signal, fs, N_fft, f0, "original signal")
    except FileNotFoundError:
        print(f"No original file named '{orig_filename}' found — skipping original signal display.")
    except Exception as e:
        print(f"Could not read original file '{orig_filename}': {e}")
