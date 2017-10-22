import numpy as np
from fft import fft

def interval_test(x):
    total_frames = len(x)
    # max_pow = np.floor(np.log2(len(x)))
    # n = 2**max_pow
    fps = 30

    lengths = [5,10,20,60]
    for length in lengths:
        frame_interval = np.floor(np.log2(fps*length))
        n = 2**frame_interval
        print("intervalo: ",n)
        for i in range(0,int(total_frames/n)):
            print(i)
            frames = total_frames[i*n,(i+1)*n - 1]

            f = np.linspace(-n / 2, n / 2 - 1, n) * fps / n
            X = np.abs(fft.FFT_shift(np.fft.fft(frames))) ** 2
            print("\tFrecuencia cardíaca: ", abs(f[np.argmax(X)]) * 60, " pulsaciones por minuto")

