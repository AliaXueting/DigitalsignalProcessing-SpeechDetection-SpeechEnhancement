import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
import librosa
import soundfile as sf

if __name__ == "__main__":

    f = wave.open("original.wav", "rb")
    parameters = f.getparams()
    print(parameters)  # Print parameters of wav file
    nchannels = parameters[0]  # Get channel parameters
    framerate = parameters[2]  # Get sampling frequency
    nframes = parameters[3]  # Get the number of sampling points
    data = f.readframes(nframes)
    f.close()
    wave_data1 = np.fromstring(data, dtype=np.short)  # convert data to decimal
    # Using Normalization processing due to the original data is a single channel
    # (there is no need to reshape the wave_data)
    wave_data1 = wave_data1 * 1.0 / (max(abs(wave_data1)))
    x1 = np.arange(0, nframes) * (1.0 / framerate)  # Horizontal and vertical time coordinates

    # Voice Enhancement
    # Calculate the spectrum of the signal, the frame length win_length, the frame shift hop_length,
    # And the number of Fourier transform points is n_fft=256 points
    originalData, fs = librosa.load("original.wav", sr=None)
    siginal = librosa.stft(originalData, n_fft=256, hop_length=128, win_length=256)  # D x T
    D, T = np.shape(siginal)
    amplitude = np.abs(siginal)
    phase = np.angle(siginal)
    power = amplitude ** 2  # get the energy spectrum of the signal
    print(fs)
    # Estimate the energy of a noisy signal
    # Since the noise signal is unknown, it is assumed that the first 20 frames of the noisy signal are noise
    amplitude_nosie = np.mean(np.abs(siginal[:, :20]), axis=1, keepdims=True)
    power_nosie = amplitude_nosie ** 2
    power_nosie = np.tile(power_nosie, [1, T])  # Repeat the first 11 frames to the same length as the noisy speech

    # smoothing method
    amplitude_new = np.copy(amplitude)
    k = 1
    for t in range(k, T - k):
        amplitude_new[:, t] = np.mean(amplitude_new[:, t - k:t + k + 1], axis=1)

    power_new = amplitude_new ** 2

    # Ultra-subtractive denoising
    alpha = 4
    gamma = 1

    Power_enhenc = np.power(power_new, gamma) - alpha * np.power(power_nosie, gamma)
    Power_enhenc = np.power(Power_enhenc, 1 / gamma)

    beta = 0.0001
    mask = (Power_enhenc >= beta * power_nosie) - 0
    Power_enhenc = mask * Power_enhenc + beta * (1 - mask) * power_nosie

    Mag_enhenc = np.sqrt(Power_enhenc)

    Mag_enhenc_new = np.copy(Mag_enhenc)
    # Calculate the maximum noise residual
    maxnr = np.max(np.abs(siginal[:, :11]) - amplitude_nosie, axis=1)

    k = 1
    for t in range(k, T - k):
        index = np.where(Mag_enhenc[:, t] < maxnr)[0]
        temp = np.min(Mag_enhenc[:, t - k:t + k + 1], axis=1)
        Mag_enhenc_new[index, t] = temp[index]

    # restore the signal
    S_enhec = Mag_enhenc_new * np.exp(1j * phase)
    enhenc = librosa.istft(S_enhec, hop_length=128, win_length=256)
    sf.write("improved.wav", enhenc, fs)

    f = wave.open("improved.wav", "rb")
    parameters = f.getparams()
    print(parameters)  # Print parameters of wav file
    nchannels = parameters[0]  # Get channel parameters
    framerate = parameters[2]  # Get sampling frequency
    nframes = parameters[3]  # Get the number of sampling points
    data2 = f.readframes(nframes)
    f.close()
    wave_data2 = np.fromstring(data2, dtype=np.short)  # convert data to decimal
    wave_data2 = wave_data2 * 1.0 / (max(abs(wave_data2)))
    x2 = np.arange(0, nframes) * (1.0 / framerate)  # Horizontal and vertical time coordinates

    ft1 = fft(wave_data1)  # It should be noted that only the data of one channel can be operated
    magnitude1 = np.absolute(ft1)
    magnitude1 = magnitude1[0:int(len(magnitude1) / 2) + 1]
    f1 = np.linspace(0, framerate, len(magnitude1))
    xfp1 = 20 * np.log10(np.clip(np.abs(magnitude1), 1e-20, 1e1000))

    ft2 = fft(wave_data2)  # It should be noted that only the data of one channel can be operated
    magnitude2 = np.absolute(ft2)
    magnitude2 = magnitude2[0:int(len(magnitude2) / 2) + 1]
    f2 = np.linspace(0, framerate, len(magnitude2))
    xfp2 = 20 * np.log10(np.clip(np.abs(magnitude2), 1e-20, 1e1000))
    xftp3 = 20 * np.log10(np.clip(np.abs(magnitude2), 1e-20, 1e1000))
    #Amplitude enhancement of signals within the highest harmonic frequency range
    for i in range(30000,34000):
        temp = xftp3[i]
        xftp3[i] = 1.5*temp    #multiply 1.5 to enchance


    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x1, wave_data1)
    plt.xlabel("Time/s", loc='right')
    plt.ylabel("Amplitude")
    plt.title("Original Signal (Time Domain)")

    plt.subplot(2, 1, 2)
    plt.plot(x2, wave_data2)
    plt.xlabel("Time/s", loc='right')
    plt.ylabel("Amplitude")
    plt.title("Enhanced Signal (Time Domain)")
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.ylabel("Amplitude")
    plt.xlabel("Time/s", loc='right')
    plt.title("Spectrogram of the Original Signal")
    plt.specgram(originalData, NFFT=256, Fs=fs)
    plt.subplot(2, 1, 2)
    plt.specgram(enhenc, NFFT=256, Fs=fs)
    plt.ylabel("Amplitude")
    plt.xlabel("Time/s", loc='right')
    plt.title("Spectrogram of the Enhanced Signal")
    plt.show()


    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(f1, magnitude1)
    plt.xscale("log")
    plt.xlabel("hz[log scale]", loc='right')
    plt.ylabel("Amplitude")
    plt.title("Original Signal--Amplitude is not Logarithmic (Frequency Domain)")

    plt.subplot(2, 1, 2)
    plt.plot(f2, magnitude2)
    plt.xscale("log")
    plt.xlabel("hz[log scale]", loc='right')
    plt.ylabel("Amplitude")
    plt.title("Enhanced Signal--Amplitude is not Logarithmic (Frequency Domain)")
    plt.show()

    plt.figure()
    plt.plot(f1, xfp1)
    plt.xscale("log")
    plt.xlabel("hz[log scale]")
    plt.ylabel("db")
    plt.title("Part of Original Signal (Frequency Domain)")
    plt.xlim((50, 18000))
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(f1, xfp1)
    plt.xscale("log")
    plt.xlabel("hz[log scale]", loc='right')
    plt.ylabel("db")
    plt.title("Original Signal--Amplitude is Logarithmic (Frequency Domain)")

    plt.subplot(2, 1, 2)
    plt.plot(f2, xfp2)
    plt.xscale("log")
    plt.xlabel("hz[log scale]", loc='right')
    plt.ylabel("db")
    plt.title("Enhanced Signal--Amplitude is Logarithmic (Frequency Domain)")
    plt.show()

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(f2, xfp2)
    plt.xscale("log")
    plt.xlabel("hz[log scale]", loc='right')
    plt.ylabel("db")
    plt.title("Enhanced Signal--Amplitude is Logarithmic (Frequency Domain")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(f2, xftp3)
    plt.xscale("log")
    plt.xlabel("hz[log scale]", loc='right')
    plt.ylabel("db")
    plt.title("Amplitude enchance for the higest frequency harmonic")
    plt.grid(True)
    plt.show()
























