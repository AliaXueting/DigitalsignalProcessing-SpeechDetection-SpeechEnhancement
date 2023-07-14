import sys
import wave
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

class VowelDetector:
    vowel_array = ["A", "E", "I", "O", "U"]
    # vowel_array = ["E"]
    def LoadFile(self, name, name_mod = True):
        if name_mod == True: 
            f = wave.open(name + ".wav", "rb")
            parameters = f.getparams()
            nframes = parameters[3]    
            data = f.readframes(nframes)
            f.close()
        else:
            f = wave.open(name, "rb")
            parameters = f.getparams()
            nframes = parameters[3] 
            data = f.readframes(nframes)
            f.close()
        return data, parameters

    def FFT(self, data, framerate):
        xf = abs(np.fft.rfft(data)) #Forier Transform
        freqs = np.linspace(0, framerate // 2, len(data) // 2 + 1)
        return xf, freqs

    def PatternMatch(self, wav, param, pattern_dic):
        wave_data = np.frombuffer(wav, dtype=np.short)  # convert into decimalism
        wave_data = wave_data * 1.0 / (max(abs(wave_data)))  # normalization
        framerate = param[2]
        xf, freqs = VowelDetector.FFT(self, wave_data, framerate) #Forier Transform
        new_peaks, new_amps = VowelDetector.PeakFinder(self, xf) #Find frequency pattern
        new_pattern = list(zip(freqs[new_peaks], new_amps['peak_heights']))
        sum_amp = sum(new_amps['peak_heights'])

        plt.plot(freqs, xf)
        plt.plot(freqs[new_peaks], xf[new_peaks], "x")
        plt.vlines(x=freqs[new_peaks], ymin=0, ymax=xf[new_peaks], colors="r")
        plt.ylabel("Amplitude")
        plt.xlabel("Frequence (Hz)")
        plt.title("Test Vowel")
        plt.show()
        
        # Match new wav file pattern with stored wav file pattern
        match_vowel = "No Match"
        fr_match_ref = 0          
        for key in pattern_dic:
            print("Key-----------------------", key)
            fr_match = 0
            key_pattern = pattern_dic[key]
            key_sum_amp = 0.1
            # Sum up all amp in key_pattern
            for pair in key_pattern: 
                key_sum_amp += pair[1]
            
            for key_pair in key_pattern:
                for new_pair in new_pattern:
                    if abs(key_pair[0] - new_pair[0]) < 10: # Compare frequency peak with a tolerance of 10Hz
                        new_amp_ratio = (new_pair[1] / sum_amp) * 100
                        key_amp_ratio = (key_pair[1] / key_sum_amp) * 100
                        if abs(key_amp_ratio - new_amp_ratio) <= 5: # Matched if corresponding peak has similar amplitude
                            print("amplitude ratio difference = ", abs(key_amp_ratio - new_amp_ratio))
                            print("match at freq: ", key_pair[0], new_pair[0])
                            fr_match += 1
            print("fr_match = ", fr_match)
            if fr_match > fr_match_ref:
               match_vowel = key
               fr_match_ref = fr_match             
                          
        return match_vowel

    def PeakFinder(self, xf):
        peaks, amps = find_peaks(xf, height=200, distance = 50, prominence= 200) #Find peaks of frequencies and corresponding amplitudes
        return peaks, amps

    def PatternFinder(self):
        Dictionary = {}
        #Find pattern for each vowel in vowel_array
        for vowel in VowelDetector.vowel_array:
            #Load wav file, and get frames and parameters
            file, param = VowelDetector.LoadFile(self, vowel, True)
            wave_data = np.frombuffer(file, dtype=np.short)  # convert into decimalism
            wave_data = wave_data * 1.0 / (max(abs(wave_data)))  # normalization
            framerate = param[2]
            
            #Generate frequency spectrum and mark patterns
            xf, freqs = VowelDetector.FFT(self, wave_data, framerate) #Forier Transform
            refine_peaks, refine_amps = VowelDetector.PeakFinder(self, xf) #Find frequency pattern

            #Plot frequency spectrum and pattern of the vowel
            plt.plot(freqs, xf)
            plt.plot(freqs[refine_peaks], xf[refine_peaks], "x")
            plt.vlines(x=freqs[refine_peaks], ymin=0, ymax=xf[refine_peaks], colors="r")
            plt.ylabel("Amplitude")
            plt.xlabel("Frequence (Hz)")
            plt.title("Training " + vowel)
            plt.show()
            
            #Generate vowel's pattern in 2D array format
            Pattern = list(zip(freqs[refine_peaks], refine_amps['peak_heights']))
                      
            #Store vowel in dictionary as key, and pattern as value
            Dictionary[vowel] = Pattern
        
        # print(Dictionary)
        return Dictionary

def voweldetector(wavfile):
    Detector = VowelDetector()

    Dic = Detector.PatternFinder()
    f, params = Detector.LoadFile(wavfile, False)
    Output = Detector.PatternMatch(f, params, Dic)
    print("MATCHED VOWEL: ", Output)
    return Output

if __name__ == "__main__":

    file_name = sys.argv[1]
    vowel = voweldetector(file_name)
