import wave
from pylab import *
from scipy.fft import fft

if __name__ == "__main__":
    f = wave.open("original.wav", "rb")
    parameters = f.getparams()
    print(parameters)   #打印wav文件相关参数
    nchannels = parameters[0]  #获取声道数
    framerate = parameters[2]  #获取采样频率
    nframes = parameters[3]    #获取采样点数
    data = f.readframes(nframes)
    f.close()
    wave_data = np.fromstring(data, dtype=np.short)  #将数据转化为十进制
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))  #归一化处理  因为原始数据是单通道 所以这里就不需要对wave_data做reshape处理
    x = np.arange(0, nframes) * (1.0 / framerate)    #横纵的时间坐标

    #以下是画的原始信号的时域图
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, wave_data)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Original Sound Signal")

    #傅里叶变换
    # xs = wave_data[:2048]    #2048这个值可以改变  做傅里叶变换的时候采样多少个点
    # xf = np.fft.rfft(xs) / 2048
    # xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e1000))
    # freqs = np.linspace(0, framerate // 2, 2048 // 2 + 1)

    ft = fft(wave_data)  # 需要注意 只能对一个通道的数据进行操作
    magnitude = np.absolute(ft)  # 取相似度
    magnitude = magnitude[0:int(len(magnitude) / 2) + 1]
    freqs = np.linspace(0, framerate, len(magnitude))
    xfp = 20 * np.log10(np.clip(np.abs(magnitude), 1e-20, 1e1000))

    #原始信号的频谱图
    plt.subplot(2, 1, 2)
    plt.plot(freqs, xfp)
    plt.xscale("log")
    plt.xlabel("Frequency (Hz)[log scale]")
    plt.ylabel("Amplitude (dB)")
    plt.title("Original Signal Spectrum")
    plt.subplots_adjust(hspace=0.4)
    plt.show()










