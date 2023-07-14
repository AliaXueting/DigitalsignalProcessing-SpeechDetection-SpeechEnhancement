
import wave
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    f = wave.open("original.wav", "rb")
    parameters = f.getparams()
    print(parameters)   #打印wav文件相关参数
    nchannels = parameters[0]  #获取声道数
    framerate = parameters[2]  #获取采样频率
    nframes = parameters[3]    #获取采样点数
    data = f.readframes(nframes)
    f.close()
    wave_data = np.frombuffer(data, dtype=np.short)  #将数据转化为十进制
    wave_data = wave_data * 1.0 / (max(abs(wave_data)))  #归一化处理  因为原始数据是单通道 所以这里就不需要对wave_data做reshape处理
    x = np.arange(0, nframes) * (1.0 / framerate)   #横纵的时间坐标

    #以下是画的原始信号的时域图 总秒数 185856/44100 = 4.2s
    plt.figure(figsize=(32, 16))
    plt.plot(x, wave_data)
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Original Sound Signal")
    plt.show()

    #1~2s时域图 （apples的时域图） 44100 * 1～ 44100*2
    plt.subplot(1, 1, 1)
    plt.plot(x[44100:88200], wave_data[44100:88200])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Original Sound Signal(0～1s)")
    plt.show()

    # 通常在0.02~0.05s这样的数量级称为一帧frame，频谱变换通常只需要一帧的波形
    # 1.2~1.24s 时域图（元音/æ/的时域图） 元音具有明显的周期性
    plt.subplot(1, 1, 1)
    plt.plot(x[52920:54684], wave_data[52920:54684])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Original Sound Signal(1.2~1.24s)")
    plt.show()
    #
    # 1.8~1.84s 时域图（辅音/z/的时域图）无周期性
    plt.subplot(1, 1, 1)
    plt.plot(x[79380:81144], wave_data[79380:81144])
    plt.xlabel("Time(s)")
    plt.ylabel("Amplitude")
    plt.title("Original Sound Signal(1.6~1.64s)")
    plt.show()

    #傅里叶变换 (1~2s)
    xs = wave_data[44100:88200]    #2048这个值可以改变  做傅里叶变换的时候采样多少个点
    xf = np.fft.rfft(xs) / 44100
    xfp = 20 * np.log10(np.clip(np.abs(xf), 1e-20, 1e100))
    freqs = np.linspace(0, framerate // 2, 44100 // 2 + 1)

    #(1~2s apples)的频谱图
    plt.figure(figsize=(32, 16))
    plt.subplot(1, 1, 1)
    plt.plot(freqs, xfp)
    plt.xlabel(u"Hz")
    plt.ylabel("db")
    plt.title("Original Signal Spectrum(1～2s)")
    plt.show()
    # #
    # 傅里叶变换 (1.2~1.24s)
    xs1 = wave_data[52920:54684]  # 2048这个值可以改变  做傅里叶变换的时候采样多少个点
    xf1 = np.fft.rfft(xs1) / 1764
    xfp1 = 20 * np.log10(np.clip(np.abs(xf1), 1e-20, 1e100))
    freqs1 = np.linspace(0, framerate // 2, 1764 // 2 + 1)

    # 1.2~1.24s 频谱图（元音/æ/的频谱图）波峰具有明显的周期性(符合元音特征)
    plt.figure(figsize=(32, 16))
    plt.subplot(1, 1, 1)
    plt.plot(freqs1, xfp1)
    plt.xlabel(u"Hz")
    plt.ylabel("db")
    plt.title("Original Signal Spectrum(1.2~1.24s)")
    plt.show()

    # 傅里叶变换 (1.8~1.84s)
    xs2 = wave_data[79380:81144]  # 2048这个值可以改变  做傅里叶变换的时候采样多少个点
    xf2 = np.fft.rfft(xs2) / 1764
    xfp2 = 20 * np.log10(np.clip(np.abs(xf2), 1e-20, 1e100))
    freqs2 = np.linspace(0, framerate // 2, 1764 // 2 + 1)

    # 1.8~1.84s 频谱图（辅音/z/的频谱图） 波峰无周期性
    plt.figure(figsize=(32, 16))
    plt.subplot(1, 1, 1)
    plt.plot(freqs2, xfp2)
    plt.xlabel(u"Hz")
    plt.ylabel("db")
    plt.title("Original Signal Spectrum(1.6~1.64s)")
    plt.show()
