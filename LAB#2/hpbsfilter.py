import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def bandstopDesign(sampling_rate, lowcut, highcut):
    taps = int(sampling_rate / 2 + 1)
    k1 = int(lowcut/sampling_rate * taps)
    k2 = int(highcut/sampling_rate * taps)
    
    X = np.ones(taps)

    X[k1 : k2+1] = 0
    X[taps-k2 : taps-k1+1] = 0
    
    # Plot frequence response
    plt.plot(X)
    plt.show()

    x = np.fft.ifft(X)
    x = np.real(x)

    h = np.zeros(taps)
    
    h[0 : int(taps/2) + 1] = x[int(taps/2) : int(taps)]
    h[int(taps/2) : int(taps)] = x[0 : int(taps/2) + 1]

    h_win = h*np.hamming(taps)

    return taps, h_win

def highpassDesign(sampling_rate, cutoff_freq):
   
    taps = int(sampling_rate / 2 + 1)
    k1 = int(cutoff_freq/sampling_rate * taps)
    
    X = np.ones(taps)
   
    X[0:int(k1+1)] = 0
    X[int(taps-k1-1):int(taps)] = 0

    # Plot frequence response
    plt.plot(X)
    plt.show()
   
    x = np.fft.ifft(X)
    x = np.real(x)
   
    h = np.zeros(taps)
    h[0 : int(taps/2)+1] = x[int(taps/2) : int(taps)]
    h[int(taps/2) : int(taps)] = x[0 : int(taps/2) + 1]

    h_win = h*np.hamming(taps)
    
    return taps, h_win

hp_taps, hp_coefs = highpassDesign(1000, 1)
hp_taps, bs_coefs = bandstopDesign(1000, 45, 55)
# print(hp_coefs)
# print(bs_coefs)
# print(len(hp_coefs))

# Plot impulse response coefficient in sample domain
plt.plot(hp_coefs)
plt.show()
plt.plot(bs_coefs)
plt.show()
