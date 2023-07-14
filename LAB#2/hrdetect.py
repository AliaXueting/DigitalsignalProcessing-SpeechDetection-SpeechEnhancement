import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hpbsfilter import bandstopDesign, highpassDesign
from firfilter import FIRfilter

if __name__ == "__main__":
    data = pd.read_csv("ECG_1000Hz_34.dat", header=None)
    df = np.array(data[0])

    fs = 1000
    taps, coefficients = bandstopDesign(fs, 45, 55)
    FIR1 = FIRfilter(taps, coefficients)
    output = FIR1.dofilter(df, 1)   # Preprocessing of the original signal

    template = output[500:1001]     # get template
    fir_coeff = template[::-1]      # get reverse template
    det = FIRfilter(taps, fir_coeff).dofilter(output, 1)  # Filter the signal with new filter coefficients
    result = det*det                # Take the square of the filtered signal

    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(df, color='green', label="Orginal ECG")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(result, color='green', label="Heart Beat Dection")
    plt.legend()


    plt.figure(2)
    plt.subplot(2, 1, 1)
    plt.plot(template, label="Template")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(fir_coeff, label="Reverse Template")
    plt.legend()


    plt.figure(3)
    plt.subplot(2, 1, 1)
    plt.plot(output, label="First Fir-filter ECG")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(det, color='green', label="match-filter ECG")
    plt.legend()
    plt.show()






