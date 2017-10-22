# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 19:23:10 2017

@author: pfierens
"""

import numpy as np
import argparse
from scipy.signal import butter, lfilter, lfilter_zi, filtfilt

import matplotlib.pyplot as plt
import cv2

from fft.fft import FFT, FFT_shift

parser = argparse.ArgumentParser(description="Photoplesthysmography")
parser.add_argument("--file", "-f", type=str, dest='file', default='/Users/natinavas/Documents/ITBA/MNA/photoplesthysmography/xid-120473_1.mp4')
parser.add_argument("--butter_filter", "-bf", type=bool, dest='filter', default=False)

args = parser.parse_args()

cap = cv2.VideoCapture(args.file)

if not cap.isOpened():
    raise Exception('Could not open video file')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

r = np.zeros((1,length))
g = np.zeros((1,length))
b = np.zeros((1,length))

k = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        r[0,k] = np.mean(frame[330:360,610:640,0])
        g[0,k] = np.mean(frame[330:360,610:640,1])
        b[0,k] = np.mean(frame[330:360,610:640,2])
    else:
        break
    k = k + 1

cap.release()
cv2.destroyAllWindows()

n = 1024
f = np.linspace(-n/2,n/2-1,n)*fps/n

r = r[0,0:n]-np.mean(r[0,0:n])
g = g[0,0:n]-np.mean(g[0,0:n])
b = b[0,0:n]-np.mean(b[0,0:n])

# B, A = butter(2, 0.12)
# r = filtfilt(B, A, r)
# g = filtfilt(B, A, g)
# b = filtfilt(B, A, b)


# zi = lfilter_zi(B,A)
# r, _ = lfilter(B, A, r, zi=zi * r[0])
# g, _ = lfilter(B, A, g, zi=zi * g[0])
# b, _ = lfilter(B, A, b, zi=zi * b[0])
# y = signal.filtfilt(b, a, xn)

print(len(r))

R = np.abs(FFT_shift(np.fft.fft(r)))**2
G = np.abs(FFT_shift(np.fft.fft(g)))**2
B = np.abs(FFT_shift(np.fft.fft(b)))**2

# Use butter filter
if args.filter:
    B_butter, A_butter = butter(2, 0.4)
    R = filtfilt(B_butter, A_butter, R)
    G = filtfilt(B_butter, A_butter, G)
    B = filtfilt(B_butter, A_butter, B)

# plt.plot(60*f,R)
# plt.xlim(0,200)
#
# plt.plot(60*f,G)
# plt.xlim(0,200)
# plt.xlabel("frecuencia [1/minuto]")
#
# plt.plot(60*f,B)
# plt.xlim(0,200)


# Compare using different colors
print("Frecuencia cardíaca R: ", abs(f[np.argmax(R)])*60, " pulsaciones por minuto")
print("Frecuencia cardíaca G: ", abs(f[np.argmax(G)])*60, " pulsaciones por minuto")
print("Frecuencia cardíaca B: ", abs(f[np.argmax(B)])*60, " pulsaciones por minuto")