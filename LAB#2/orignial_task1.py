import numpy as np
import matplotlib.pyplot as plt


class FIRFilter:
    def __init__(self):
       pass

    def bandstopDesign(self, sampling_rate, lowcut, highcut):
        taps = int(sampling_rate / 2 + 1)
        k1 = int(lowcut / sampling_rate * taps)
        k2 = int(highcut / sampling_rate * taps)

        X = np.ones(taps)

        X[k1: k2 + 1] = 0
        X[taps - k2: taps - k1 + 1] = 0

        x = np.fft.ifft(X)
        x = np.real(x)

        return taps, X, x

    def highPassFilter(self, sampling_rate, cut):
        taps = int(sampling_rate / 2 + 1)
        k1 = int(cut / sampling_rate * taps)

        X = np.zeros(taps)

        X[k1:] = 1
        X[taps - k1:] = 0

        x = np.fft.ifft(X)
        x = np.real(x)

        return taps, X, x

if __name__ == "__main__":
    filter = FIRFilter()
    N, X, x = filter.bandstopDesign(1000, 45, 55)
    fig2, ax = plt.subplots(nrows=2)
    fig2.set_size_inches(12, 8)
    fig2.suptitle('Bandstop Filter: Sampling Domain & Frequency Domain', fontsize=12)
    ax[1].plot(X, label='Frequency Domain', color="blue")
    ax[1].legend()
    ax[0].plot(x, label='Sampling Domain', color="blue")
    ax[0].legend()
    plt.show()














