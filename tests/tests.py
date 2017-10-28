import numpy as np
from fft import fft

def interval_test(x,fps):
    total_frames = len(x)

    # amount of interval's frames
    ns = [512,1024,2048,4096]
    for n in ns:
        print("intervalo: ",n)
        for i in range(0,int(total_frames/n)):
            if total_frames > n:
                print(i)
                frames = x[i*n:(i+1)*n - 1]

                f = np.linspace(-n / 2, n / 2 - 1, n) * fps / n
                X = np.abs(fft.FFT_shift(np.fft.fft(frames))) ** 2
                print("\tFrecuencia cardíaca: ", abs(f[np.argmax(X)]) * 60, " pulsaciones por minuto")

