import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from hpbsfilter import bandstopDesign, highpassDesign

class FIRfilter:
    def __init__(self, _taps, _coefficients):
        self._coefficients = _coefficients
        self._taps = _taps
    
    def dofilter(self, data, method):
        filtereddata = np.zeros(len(data))
        buffer = np.zeros(self._taps)
        offset = int(self._taps / 2) #set offset at M/2, can be other position too
        
        if method == 1:
            #Quick Implementation -- Ring Buffer Method
            for i in range(len(data)):
                buffer[offset] = data[i] #insert new data point
                buff_index = offset
                coef_index = 0
                output = 0
                
                #Multiplication from offset position to the end of the delay line
                while buff_index < self._taps:
                    output += buffer[buff_index] * self._coefficients[coef_index]
                    buff_index += 1
                    coef_index += 1
                
                #Multiplication from the beginning of the array to the offset position
                loop_index = 0
                while coef_index < self._taps:
                    output += buffer[loop_index] * self._coefficients[coef_index]
                    loop_index += 1
                    coef_index += 1
            
                #Move offset pointer to the left
                offset -= 1
                
                #Loop around when point reached beginning of the array
                if offset == 0:
                    offset = self._taps - 1
                
                filtereddata[i] = output
        elif method == 2:        
            #Slow Implementation -- Move Each Element at Every Time Step
            for i in range(len(data)):
                for j in range(self._taps - 1, 0, -1):
                    buffer[j] = buffer[j-1]
                
                buffer[0] = data[i]
                output = 0

                for k in range(0, self._taps, 1):
                    output += buffer[k] * self._coefficients[k]

                filtereddata[i] = output
        else: 
            print('no method selected')
        
        return filtereddata       

if __name__ == '__main__':
    data = pd.read_csv("ECG_1000Hz_34.dat", header=None)
    # plt.plot(data)
    # plt.show()
    df = np.array(data[0])

    fs = 1000
    hp_taps, hp_coefficients = highpassDesign(fs, 1)
    bs_taps, bs_coefficients = bandstopDesign(fs, 45, 55)

    FIR_hp = FIRfilter(hp_taps, hp_coefficients)
    FIR_bs = FIRfilter(bs_taps, bs_coefficients)

    hp_output = FIR_hp.dofilter(df, 1)
    bs_output = FIR_bs.dofilter(hp_output, 1)
        
    plt.figure(1)
    plt.plot(df,label="Orginal ECG", color = 'red')
    plt.plot(hp_output,label="Filtered ECG w/o DC", color = 'orange')
    plt.legend()
    plt.show()

    plt.figure(2)
    plt.plot(df,label="Orginal ECG", color = 'red')
    plt.plot(bs_output,label="Filtered ECG w/o DC w/o 50 hz", color = 'orange')
    plt.legend()
    plt.show()