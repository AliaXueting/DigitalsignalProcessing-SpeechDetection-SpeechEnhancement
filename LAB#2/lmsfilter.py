import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class FIRfilter:
    def __init__(self, _cofficients, _taps):
        self.cofficients = _cofficients
        self.taps = _taps
        self.buffer = np.zeros(self.taps)

    def filter(self, v):
        for j in range(0,self.taps-1):
            self.buffer[self.taps-j-1] = self.buffer[self.taps-j-2]

        self.buffer[0] = v
        result = np.inner(self.buffer, self.cofficients)

        return result

    def doFilterAdaptive(self, signal, noise, learningRate):
        length = len(signal)
        y_predict = np.empty(length)
        for i in range(0, length):
            temp = fir_filter.filter(noise[i])
            output = signal[i] - temp  # error
            for j in range(self.taps):
                # Formula of gradient descent algorithm Update filter coefficients
                self.cofficients[j] = self.cofficients[j] + output * learningRate * self.buffer[j]
            y_predict[i] = output

        return y_predict



if __name__ == "__main__":
    df = pd.read_csv("ECG_1000Hz_34.dat", header=None)
    data = df[0].tolist()

    length = len(data)
    signal = np.array(data)
    noise_fre = 50
    fs = 1000
    number_taps = 200
    learning_rate = 0.0001

    fir_filter = FIRfilter(np.zeros(number_taps), number_taps)
    noise = np.sin(np.linspace(0, noise_fre*2*np.pi/fs*length,length))  # Define 50Hz noise
    y_predict = fir_filter.doFilterAdaptive(signal, noise, learning_rate)  # Signal output from LMS
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(signal, color='green', label="Orginal ECG")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(y_predict, color='green', label="LMS Filtered ECG")
    plt.legend()
    plt.show()






