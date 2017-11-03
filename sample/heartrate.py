# -*- coding: utf-8 -*-

import numpy as np
import argparse
import cv2
from scipy.signal import butter, filtfilt
from fft.fft import fft_shift, fft

# Read program arguments
parser = argparse.ArgumentParser(description="Photoplesthysmography")
parser.add_argument("--file", "-f", type=str, dest='file', default=None)
parser.add_argument("--butter_filter", "-bf", type=bool, dest='filter', default=True)
parser.add_argument("--fft_type", "-ft", type=str, dest='fft', default='iterative')
parser.add_argument("--color_channel", "-cc", type=str, dest='color', default="RGB")
parser.add_argument("--interval", "-i", type=int, dest="interval", default=1024)
parser.add_argument("--min_height", "-minh", type=int, dest="min_height", default=610)
parser.add_argument("--max_height", "-maxh", type=int, dest="max_height", default=640)
parser.add_argument("--min_width", "-minw", type=int, dest="min_width", default=330)
parser.add_argument("--max_width", "-maxw", type=int, dest="max_width", default=360)

args = parser.parse_args()

file = args.file

cap = cv2.VideoCapture(file)

if not cap.isOpened():
    raise Exception('Could not open video file')

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Initialize
if 'R' in args.color:
    r = np.zeros((1,length))
if 'G' in args.color:
    g = np.zeros((1,length))
if 'B' in args.color:
    b = np.zeros((1,length))

maxWidth = args.max_width
minWidth = args.min_width
maxHeight = args.max_height
minHeight = args.min_height

if maxHeight > height or maxWidth > width or minHeight < 0 or minWidth < 0:
    raise Exception("The provided dimensions are not valid")

k = 0

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        if 'B' in args.color:
            b[0,k] = np.mean(frame[minWidth:maxWidth,minHeight:maxHeight,0])
        if 'G' in args.color:
            g[0,k] = np.mean(frame[minWidth:maxWidth,minHeight:maxHeight,1])
        if 'R' in args.color:
            r[0,k] = np.mean(frame[minWidth:maxWidth,minHeight:maxHeight,2])
    else:
        break
    k = k + 1

cap.release()
cv2.destroyAllWindows()

n = args.interval
f = np.linspace(-n/2,n/2-1,n)*fps/n

# Normalize vectors
if 'R' in args.color:
    r = r[0,0:n]-np.mean(r[0,0:n])
if 'G' in args.color:
    g = g[0,0:n]-np.mean(g[0,0:n])
if 'B' in args.color:
    b = b[0,0:n]-np.mean(b[0,0:n])

# Use butter filter
if args.filter:
    BPM_lowest = 40
    BPM_max = 230
    Wn = [((BPM_lowest/60)/fps * 2), ((BPM_max/60)/fps * 2)]
    B_butter, A_butter = butter(N=2, Wn=Wn, btype='band')
    if 'R' in args.color:
        r = filtfilt(B_butter, A_butter, r)
    if 'G' in args.color:
        g = filtfilt(B_butter, A_butter, g)
    if 'B' in args.color:
        b = filtfilt(B_butter, A_butter, b)

# Calculate fourier transformations
if 'R' in args.color:
    R = np.abs(fft_shift(fft(r, args.fft))) ** 2
if 'G' in args.color:
    G = np.abs(fft_shift(fft(g, args.fft))) ** 2
if 'B' in args.color:
    B = np.abs(fft_shift(fft(b, args.fft))) ** 2

# Compare using different colors
if 'R' in args.color:
    print("Frecuencia cardíaca R: ", abs(f[np.argmax(R)])*60, " pulsaciones por minuto")
if 'G' in args.color:
    print("Frecuencia cardíaca G: ", abs(f[np.argmax(G)])*60, " pulsaciones por minuto")
if 'B' in args.color:
    print("Frecuencia cardíaca B: ", abs(f[np.argmax(B)])*60, " pulsaciones por minuto")