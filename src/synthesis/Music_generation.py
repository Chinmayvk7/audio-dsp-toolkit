import numpy as np
import soundfile as sf

# Sampling rate
fs = 8000  

# Time array: each note is 0.5 seconds
t = np.arange(0, 0.5, 1 / fs)

# MIDI note numbers for Twinkle Twinkle
notes = np.array([
    52, 52, 59, 59, 61, 61, 59, 59,
    57, 57, 56, 56, 54, 54,
    56, 52, 59, 57, 57, 56, 56, 54, 54
])

length = notes.size

# Generate first note
f = 440 * (2 ** ((notes[0] - 49) / 12))
print(f"number = {notes[0]}, frequency = {f:.2f}")
output = np.cos(2 * np.pi * f * t)

# Generate remaining notes
i = 1
while i < length:
    f = 440 * (2 ** ((notes[i] - 49) / 12))
    print(f"number = {notes[i]}, frequency = {f:.2f}")
    x = np.cos(2 * np.pi * f * t)
    output = np.concatenate((output, x))
    i += 1


sf.write("twinkle.wav", output, fs)

print("\n Audio file 'twinkle.wav' generated successfully!")
print(" Play it using any media player or:")
print("   Windows: start twinkle.wav")
print("   macOS:   afplay twinkle.wav")
print("   Linux:   aplay twinkle.wav")
