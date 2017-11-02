import cv2
import numpy as np
from fft import fft
from scipy.signal import butter, filtfilt

#TODO CAMBIAR FFT CALL

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
                X = np.abs(fft.FFT_shift(fft.FFT(frames))) ** 2
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

def time_test(x,window,step,fps,filename):
    total_frames = len(x)

    for i in range(0,int((total_frames - window) / step)):
        frames = x[i * step:(i * step) + window]
        f = np.linspace(-window / 2, window / 2 - 1, window) * fps / window
        X = np.abs(fft.FFT_shift(np.fft.fft(frames))) ** 2
        freq = abs(f[np.argmax(X)]) * 60
        print("Tiempo: ",((i * step) + window)/fps,"\tFrecuencia: ",freq)
        file = open(filename, "w" if i == 0 else "a")
        file.write("%g,%g\n" % (((i * step) + window)/fps, freq))
        file.close()

def subframes_test(file, interval, rows, cols, fft_method, butter_filter = True):
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print("No lo pude abrir")

    firstFrames = 50
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    r = np.zeros((rows * cols, interval))
    g = np.zeros((rows * cols, interval))
    b = np.zeros((rows * cols, interval))

    subframe_height = int(height / rows)

    subframe_width = int(width / cols)

    k = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        for i in range(0,rows):
            for j in range(0, cols):
                if ret == True:
                    if k >= firstFrames:
                        r[i * cols + j, k - firstFrames] = np.mean(frame[i*subframe_height:(i+1)*subframe_height-1,
                                                                   j * subframe_width:(j + 1) * subframe_width - 1, 0])
                        g[i * cols + j, k - firstFrames] = np.mean(frame[i*subframe_height:(i+1)*subframe_height-1,
                                                                   j * subframe_width:(j + 1) * subframe_width - 1, 1])
                        b[i * cols + j, k - firstFrames] = np.mean(frame[i*subframe_height:(i+1)*subframe_height-1,
                                                                   j * subframe_width:(j + 1) * subframe_width - 1, 2])

                else:
                    break

        k = k + 1
        if ret == False or k >= interval + firstFrames:
            break

    cap.release()
    cv2.destroyAllWindows()

    for i in range(0, rows):
        for j in range(0, cols):
            r[i * cols + j] = r[i * cols + j, 0:interval] - np.mean(r[i * cols + j, 0:interval])
            g[i * cols + j] = g[i * cols + j, 0:interval] - np.mean(g[i * cols + j, 0:interval])
            b[i * cols + j] = b[i * cols + j, 0:interval] - np.mean(b[i * cols + j, 0:interval])

    # Use butter filter
    if butter_filter:
        BPM_lowest = 40
        BPM_max = 230
        Wn = [((BPM_lowest / 60) / fps * 2), ((BPM_max / 60) / fps * 2)]
        B_butter, A_butter = butter(N=2, Wn=Wn, btype='band')
        r = filtfilt(B_butter, A_butter, r)
        g = filtfilt(B_butter, A_butter, g)
        b = filtfilt(B_butter, A_butter, b)

    for i in range(0, rows):
        for j in range(0, cols):
            f = np.linspace(-interval / 2, interval / 2 - 1, interval) * fps / interval
            R = np.abs(fft.FFT_shift(fft_method((r[i * cols + j, 0:interval])))) ** 2
            G = np.abs(fft.FFT_shift(fft_method((g[i * cols + j, 0:interval])))) ** 2
            B = np.abs(fft.FFT_shift(fft_method((b[i * cols + j, 0:interval])))) ** 2
            freq_R = abs(f[np.argmax(R)]) * 60
            freq_G = abs(f[np.argmax(G)]) * 60
            freq_B = abs(f[np.argmax(B)]) * 60

            print("row: ", i, " - col: ", j)
            print("\tR: ", freq_R, "\tG: ", freq_G, "\tB: ", freq_B)

def size_test(file, interval, fft_method, steps, butter_filter = True):
    cap = cv2.VideoCapture(file)

    if not cap.isOpened():
        print("No lo pude abrir")

    firstFrames = 50
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step_width = width / (2 * steps)
    step_height = height / (2 * steps)

    r = np.zeros((steps, interval))
    g = np.zeros((steps, interval))
    b = np.zeros((steps, interval))

    k = 0
    while (cap.isOpened()):
        ret, frame = cap.read()

        for i in range(0,steps):
            if ret == True:
                if k >= firstFrames:
                    r[i, k - firstFrames] = np.mean(
                            frame[int(height/2 - step_height*(i+1)):int(height/2 + step_height*(i+1))-1,
                            int(width/2 - step_width*(i+1)):int(width/2 + step_width*(i+1))-1, 0])
                    g[i, k - firstFrames] = np.mean(
                        frame[int(height / 2 - step_height * (i+1)):int(height / 2 + step_height * (i+1))-1,
                        int(width / 2 - step_width * (i+1)):int(width / 2 + step_width * (i+1))-1, 1])
                    b[i, k - firstFrames] = np.mean(
                        frame[int(height / 2 - step_height * (i+1)):int(height / 2 + step_height * (i+1))-1,
                        int(width / 2 - step_width * (i+1)):int(width / 2 + step_width * (i+1))-1, 2])
        k = k + 1
        if ret == False or k >= interval + firstFrames:
            break

    cap.release()
    cv2.destroyAllWindows()

    for i in range(0,steps):
        r[i] = r[i, 0:interval] - np.mean(r[i, 0:interval])
        g[i] = r[i, 0:interval] - np.mean(g[i, 0:interval])
        b[i] = r[i, 0:interval] - np.mean(b[i, 0:interval])

    # Use butter filter
    if butter_filter:
        BPM_lowest = 40
        BPM_max = 230
        Wn = [((BPM_lowest / 60) / fps * 2), ((BPM_max / 60) / fps * 2)]
        B_butter, A_butter = butter(N=2, Wn=Wn, btype='band')
        r = filtfilt(B_butter, A_butter, r)
        g = filtfilt(B_butter, A_butter, g)
        b = filtfilt(B_butter, A_butter, b)

    for i in range(0,steps):
        f = np.linspace(-interval / 2, interval / 2 - 1, interval) * fps / interval
        R = np.abs(fft.FFT_shift(fft_method((r[i, 0:interval])))) ** 2
        G = np.abs(fft.FFT_shift(fft_method((g[i, 0:interval])))) ** 2
        B = np.abs(fft.FFT_shift(fft_method((b[i, 0:interval])))) ** 2
        freq_R = abs(f[np.argmax(R)]) * 60
        freq_G = abs(f[np.argmax(G)]) * 60
        freq_B = abs(f[np.argmax(B)]) * 60

        print("Step ",i)
        print("\tR: ", freq_R, "\tG: ", freq_G, "\tB: ", freq_B)

