import pyaudio as pa
import wave
import matplotlib.pyplot as plt
from scipy.fftpack import fft
from scipy.io import wavfile
from scipy.io import wavfile as wav
import numpy as np

## Audio Input ##

CHUCK = 1024
FORMAT = pa.paInt16
CHANNELS = 2
RATE = 44100
RECORD_SECONDS = 5
WAVE_OUTPUT_FILENAME = "output.wav"

p = pa.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUCK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUCK * RECORD_SECONDS)):
    data = stream.read(CHUCK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

## Fourier ##

rate, sound = wavfile.read('output.wav')
sound = sound / 2.0**15

plt.subplot(2,1,1)
plt.plot(sound[:,0], 'r')
plt.xlabel("left channel, sample #")
plt.subplot(2,1,2)
plt.plot(sound[:,1], 'b')
plt.xlabel("right channel, sample #")
plt.tight_layout()
plt.show() ## Plot the waveform

length_in_s = sound.shape[0] / rate
signal = sound[:,0]
time = np.arange(sound.shape[0]) / sound.shape[0] * length_in_s
plt.plot(time[6000:7000], signal[6000:7000])
plt.xlabel("time, s")
plt.ylabel("Signal, relative units")
plt.show() ## Plot the signal (as if received by ear membrane)

fft_spectrum = np.fft.rfft(signal) ## fast fourier transform
freq = np.fft.rfftfreq(signal.size, d=1./rate)

fft_spectrum_abs = np.abs(fft_spectrum) ## has imaginary numbers so we take the absolute value (also shows peaks)

plt.plot(freq, fft_spectrum_abs)
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show()  ## Plot the FFT spectrum

plt.plot(freq[3000:30000], fft_spectrum_abs[3000:30000])
plt.xlabel("frequency, Hz")
plt.ylabel("Amplitude, units")
plt.show() ## Plot the FFT spectrum with only a slice