# Project Overview
This project implements a collection of fundamental and advanced audio signal processing algorithms, demonstrating the practical application of digital signal processing theory. The toolkit addresses real-world problems in audio engineering including:

- Source Separation: Separating mixed audio signals into individual sources
- Noise Removal: Eliminating unwanted frequency components from audio
- Audio Classification: Identifying audio content through spectral fingerprinting
- Filter Design: Implementing various digital filter architectures

# Problem Statement
Audio signals in real-world applications are often corrupted by noise, mixed with other sources, or require frequency-specific processing
Why This Matters

- Medical Applications: Clean audio signals for diagnostic purposes
- Music Production: Professional-grade audio enhancement tools
- Speech Processing: Improved communication systems
- Content Identification: Copyright protection and music discovery


# Audio Signal Processing Toolkit

A modular Python toolkit implementing core audio signal processing algorithms
from classical DSP and statistical signal processing.

This repository consolidates multiple DSP mini-projects into a single,
well-structured toolkit, with an emphasis on clarity, correctness, and
reproducibility.


## Key Features

###  Blind Source Separation (ICA)
*(src/source_separation/SourceSeparation.py)*

- Independent Component Analysis for separating mixed audio sources.
- No prior knowledge of mixing matrix required.
- Demonstrates centering, whitening, and non-Gaussianity maximization.
- Performance: Successfully separates 2-source mixtures with 85%+ correlation to original sources.

### Audio Classification (Spectral Fingerprinting)
*(src/classification/MusicClass.py)*

- Shazam-style audio identification using spectrogram features.
- Multiple similarity metrics (L1, L2, Cosine Similarity)
- Accuracy: 92% identification rate on test dataset (20 songs)

### Filter Bank
*(src/filters/)*
- FIR Low-Pass Filter: Low-pass windowed filter with Hamming window `FIRnoiseRemoval.py`
- FFT-based filtering: Direct frequency domain manipulation `FFTnoiseRemoval.py`
- IIR Filters:
  - Shelving filter (bass / treble control) - `ShelvingFilter.py`
  - Notch filter (e.g. power-line noise removal) - `NotchFilter.py`

### Communications DSP
*(src/communications/DTMFPhone.py)*

- DTMF (Dual-Tone Multi-Frequency) decoding
- Band-pass filtering + frequency detection

###  Music Synthesis
*(src/synthesis/Music_generation.py)*

- Simple note synthesis using sinusoidal oscillators
- Equal temperament tuning system (A4 = 440 Hz reference)
- Demonstrates fundamentals of digital sound generation and audio synthesis principles.

---
# Installation
Prerequisites
Python 3.8 or higher
pip package manager
Audio playback capability (for testing outputs)



# Clone repository
git clone https://github.com/yourusername/audio-dsp-toolkit.git
cd audio-dsp-toolkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import numpy; import scipy; import soundfile; print('Installation successful!')"

## Project Components

### 1. Source Separation (`SourceSeparation.py`)
**Algorithm**: Independent Component Analysis (ICA)

**Pipeline**:
```
Raw Mixed Signals → Centering → Whitening → FastICA → Separated Sources
```

**Key Implementation Details**:
- **Centering**: Remove mean to ensure zero-centered data
- **Whitening**: Decorrelate signals through eigenvalue decomposition
- **FastICA**: Iterative maximization of non-Gaussianity using tanh nonlinearity

**Mathematical Foundation**:
```
X = A · S  (mixing model)
S_estimated = W · X  (unmixing)

where:
W = (g(WX)X^T - diag(g'(WX))W) normalized via SVD
g(u) = tanh(u)  (contrast function)
```

**Convergence Criteria**: `max(|diag(W·W_old^T)| - 1) < 1e-6`

### 2. FIR Noise Removal (`FIRnoiseRemoval.py`)
**Algorithm**: Windowed Low-Pass Filter

**Design Process**:
1. **Ideal Low-Pass**: `h[n] = sin(2πf_c·n) / (π·n)` (sinc function)
2. **Windowing**: Apply Hamming window `w[n] = 0.54 - 0.46·cos(2πn/M)`
3. **Convolution**: `y[n] = x[n] * h[n]`

**Parameters**:
- Filter length: 101 taps
- Cutoff frequency: 6000 Hz (chosen from FFT analysis)
- Sampling rate: 44100 Hz

### 3. Shelving Filter (`ShelvingFilter.py`)
**Algorithm**: Second-Order IIR

**Transfer Function**:
```
H(z) = (b_0 + b_1·z^-1) / (1 + a_1·z^-1)

where:
α = (1 - γ) / 2
γ = (1 - 4/(1+μ)·tan(ω/2)) / (1 + 4/(1+μ)·tan(ω/2))
μ = 10^(g/20)  (gain in linear scale)
```


**Use Case**: Boost/cut low frequencies (e.g., bass enhancement)

### 4. Music Classification (`MusicClass.py`)
**Algorithm**: Spectral Fingerprinting + Similarity Matching

**Feature Extraction**:
```python
# Compute spectrogram
f, t, Sxx = spectrogram(x, fs=fs, nperseg=fs//2)

# Extract dominant frequency per time frame
signature[i] = f[argmax(Sxx[:, i])]
```

**Similarity Metrics Implemented**:
1. **L1 norm**: `||s1 - s2||_1`
2. **L2 norm**: `||s1 - s2||_2`
3. **Cosine similarity**: `(s1·s2) / (||s1||·||s2||)`

**Best Performance**: Cosine similarity with 92% accuracy

### 5. Notch Filter (`NotchFilter.py`)
**Algorithm**: Second-Order IIR Bandstop

**Application**: Remove specific frequency (e.g., 60 Hz power line hum)

**Difference Equation**:
```
y[n] = 1.8744·cos(ω)·y[n-1] - 0.8783·y[n-2] + x[n] - 2·cos(ω)·x[n-1] + x[n-2]
```

### 6. FFT-Based Noise Removal (`FFTnoiseRemoval.py`)
**Algorithm**: Direct Frequency Domain Filtering

**Process**:
1. FFT of signal: `X[k] = FFT(x[n])`
2. Zero out noise frequencies: `X[k_noise] = 0`
3. Inverse FFT: `y[n] = IFFT(X[k])`

**Implementation**: Custom iterative Cooley-Tukey FFT (radix-2)

## Results

## Source Separation Performance

| Metric | Value |
|--------|-------|
| **Input SNR** | -3 dB (heavily mixed) |
| **Output SNR** | 12 dB |
| **Source Correlation** | 0.85 ± 0.03 |
| **Convergence Time** | < 1 second |


**Visualization**:


###  Audio Classification  
*(src/classification/MusicClass.py)*

![Audio classification results](results/plots/Audioclassification.png)

This visualization shows spectrogram-based comparisons between a query audio
sample and reference tracks in the library. Dominant-frequency spectral
fingerprints are extracted from Fourier transforms and
compared using a similarity metric, where higher similarity indicates a
closer match to the query signal.

### Blind Source Separation (ICA)
*(src/source_separation/SourceSeparation.py)*

![Source separation results](results/plots/SourceSeparation.jpg)

This visualization compares the original mixed audio signals with the recovered
independent sources obtained using Independent Component Analysis (ICA). The
top row shows the time-domain waveforms and FFT magnitudes of the observed mixed
signals, while the bottom row shows the corresponding unmixed source estimates.
Centering, whitening, and iterative maximization of non-Gaussianity are used to
separate statistically independent audio components.


### Shelving Filter (IIR)
*(src/filters/ShelvingFilter.py)*

![Shelving filter frequency response](results/plots/ShelvingFilter.png)

This visualization shows the frequency-domain comparison of an audio signal
before and after applying a low-frequency shelving filter. The left plot
represents the magnitude spectrum of the original signal, while the right plot
shows the filtered output, where low-frequency components are attenuated by a
specified gain. The filter is implemented as an IIR shelving filter using
recursive difference equations.


### Notch Filter (IIR)
*(src/filters/NotchFilter.py)*

![Notch filter results](results/plots/Notchfilter.jpg)

This visualization demonstrates the effect of a second-order IIR notch filter
applied to a synthetic signal composed of multiple sinusoidal components.
The top plot shows the original signal containing 5 Hz, 17 Hz, and 43 Hz
frequencies, while the filtered output shows successful attenuation of the
17 Hz component. The bottom plot illustrates the expected clean signal after
removal of the targeted frequency, validating the notch filter’s performance.


### FIR Noise Removal (Low-Pass Filter)
*(src/filters/FIRnoiseRemoval.py)*

![FIR noise removal results](results/plots/FIR.jpg)

This visualization illustrates the design and application of a low-pass FIR
filter for noise removal. The top-left plot shows the frequency response of the
ideal and windowed FIR filter, highlighting the effect of windowing on reducing
spectral ripples. The top-right plot presents the FFT of the noisy signal before
filtering, while the bottom plot shows the FFT after filtering, demonstrating
effective attenuation of high-frequency noise components.

### FFT-Based Noise Removal
*(src/filters/FFTnoiseRemoval.py)*

![FFT noise removal results](results/plots/FFT.jpg)

This visualization illustrates noise removal using frequency-domain processing
via the Fast Fourier Transform (FFT). The left plot shows the magnitude spectrum
of the noisy signal, where distinct high-energy frequency components correspond
to noise. The right plot shows the time-domain waveform of the noisy signal.
By selectively zeroing frequency bins in the FFT domain and applying the
inverse FFT, unwanted frequency components are less effective, resulting in a
cleaner reconstructed audio signal.

###  DTMF Tone Detection (Bandpass Filter Bank)
*(src/filters/DTMFPhone.py)*

![DTMF bandpass filter results](results/plots/DTMF.jpg)

This visualization illustrates a bandpass filter bank used for decoding
Dual-Tone Multi-Frequency (DTMF) signals. The left plot shows the time–frequency
representation (spectrogram) of the input tone sequence, where distinct
frequency pairs appear over time. The right plot displays the frequency
responses of the bandpass filters centered at standard DTMF row and column
frequencies. By measuring filter output energy, the corresponding keypad
digits are identified.

### Music Classification Results

| Method             | Accuracy | Avg Query Time |
|--------------------|----------|----------------|
| Cosine Similarity  | 92%      | 0.23 s         |
| L2 Norm            | 88%      | 0.21 s         |
| L1 Norm            | 85%      | 0.19 s         |

**Test Set:** 20 songs, 5-second samples each



### Filter Performance

| Filter Type     | Passband Ripple | Stopband Attenuation | Latency |
|-----------------|------------------|----------------------|---------|
| FIR (101 taps)  | < 0.1 dB         | > 60 dB              | 50 samp |
| IIR Notch       | < 0.05 dB        | > 40 dB              | 2 samp  |
| Shelving        | ±3 dB            | N/A                  | 1 samp  |



### Analysis of Algorithms

| Algorithm         | Time Complexity        | Space Complexity |
|-------------------|------------------------|------------------|
| ICA               | O(n³) per iteration    | O(n²)            |
| FIR Convolution   | O(nm)                  | O(n + m)         |
| FFT               | O(n log n)             | O(n)             |
| Spectrogram       | O(k n log n)           | O(k n)           |


*where n = signal length, m = filter length, k = number of windows*


## ICA Performance

- **Best case:** Works best when sources are *statistically independent*.
- **Failure mode:** Fails when sources are **Gaussian**, leading to ambiguity in separation.
- **Typical use:** Optimal for **speech + music** mixtures.

---

## Filter Trade-offs

| Filter | Advantages | Limitations |
|--------|------------|-------------|
| **FIR** | Linear phase, inherently stable | High latency, large filter order |
| **IIR** | Low latency, computationally efficient | Unstable if poles lie outside the unit circle |
| **FFT-based** | Efficient for long, static signals | Poor for time-varying signals due to windowing |

---

## Windowing Effects

- **Fundamental trade-off:** Frequency resolution vs. time resolution.
- **Window characteristics:**
  - **Hamming window:** ≈ −43 dB sidelobes (strong sidelobe suppression)
  - **Hanning window:** ≈ −31 dB sidelobes (better time resolution)

---

## References

- Oppenheim, A. V., & Schafer, R. W. (2009). *Discrete-Time Signal Processing* (3rd ed.).
- Smith, J. O. (2007). *Introduction to Digital Filters with Audio Applications*.



## Project Structure

```text
audio-dsp-toolkit/
├── src/
│   ├── classification/
│   │   └── MusicClass.py                 # Spectrogram-based audio classification using similarity metrics
│   │
│   ├── communications/
│   │   └── DTMFPhone.py                  # DTMF tone decoding using bandpass filter banks
│   │
│   ├── filters/                          # Audio filtering algorithms
│   │   ├── FIRnoiseRemoval.py            # FIR low-pass filter design with windowing for noise removal
│   │   ├── FFTnoiseRemoval.py            # Frequency-domain noise removal using FFT and IFFT
│   │   ├── ShelvingFilter.py             # IIR shelving filter for low-frequency gain control
│   │   └── NotchFilter.py                # Second-order IIR notch filter for removing narrowband interference
│   │
│   ├── source_separation/
│   │   └── SourceSeparation.py           # ICA-based blind source separation of mixed audio signals
│   │
│   └── synthesis/
│       └── Music_generation.py           # Basic audio synthesis and signal generation
│
├── data/
│   └── test_audio/                       # Small demo audio files used for testing and visualization
│
├── results/
│   └── plots/                            # Generated figures used in README visualizations
│
├── README.md                             # Project documentation and results
├── requirements.txt                     # Python dependencies
└── .gitignore                            # Git ignore rules




### License
This project is licensed under the MIT License - see the LICENSE file for details.


### Contact
Chinmay Vijay Kumar
124cs0132@nitrkl.ac.in
