import numpy as np
from fft import fft

def interval_test(x,fps):
    total_frames = len(x)
    # amount of interval's frames
    ns = [512,1024,2048,4096]
    #info is a n dimensional array with avg, std, max, min in first 4
    info = np.zeros((4, len(ns)))

    for i in range(0,len(ns)):
        n = ns[i]
        print("intervalo: ",n)
        freqs = np.zeros(int(total_frames/n))
        for j in range(0,int(total_frames/n)):
            if total_frames > n:
                frames = x[j*n:(j+1)*n]

                f = np.linspace(-n / 2, n / 2 - 1, n) * fps / n
                X = np.abs(fft.FFT_shift(np.fft.fft(frames))) ** 2
                freq = abs(f[np.argmax(X)]) * 60
                print("\tFrecuencia cardíaca: ", freq, " pulsaciones por minuto")
                freqs[j]=freq

        if len(freqs)!=0:
            info[0,i] = np.mean(freqs)
            info[1,i] = np.std(freqs)
            info[2,i] = np.max(freqs)
            info[3,i] = np.min(freqs)


    print('-----------')

    for i in range(0, len(ns)):
        print("Intervalo ", ns[i], ": ")
        print("\tMedia: ", info[0][i])
        print("\tDesvío: ", info[1][i])
        print("\tMáximo: ", info[2][i])
        print("\tMínimo: ", info[3][i])
        print("\tAmplitud: ", info[2][i]-info[3][i])
