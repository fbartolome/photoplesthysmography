import numpy as np
import tests
import argparse
import cv2
from scipy.signal import butter, filtfilt

parser = argparse.ArgumentParser(description="Photoplesthysmography")
#parser.add_argument("--file", "-f", type=str, dest='file', default='/Users/natinavas/Documents/ITBA/MNA/photoplesthysmography/xid-120473_1.mp4')
parser.add_argument("--file", "-f", type=str, dest='file', default='/Users/natinavas/Downloads/video4minSeba.MOV')

args = parser.parse_args()

cap = cv2.VideoCapture(args.file)

if not cap.isOpened():
   print("No lo pude abrir")

firstFrames = 50
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - firstFrames
print(length)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(width)
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(height)
fps = cap.get(cv2.CAP_PROP_FPS)

r = np.zeros((1, length))
g = np.zeros((1, length))
b = np.zeros((1, length))

minX = 0
maxX = 1000
minY = 0
maxY = 700

k = 0
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True :
        if k >= firstFrames:
            r[0, k - firstFrames] = np.mean(frame[minX:maxX, minY:maxY, 0])
            g[0, k - firstFrames] = np.mean(frame[minX:maxX, minY:maxY, 1])
            b[0, k - firstFrames] = np.mean(frame[minX:maxX, minY:maxY, 2])
    # print(k)
    else:
        break
    k = k + 1

cap.release()
cv2.destroyAllWindows()

r = r[0,0:length]-np.mean(r[0,0:length])
g = g[0,0:length]-np.mean(g[0,0:length])
b = b[0,0:length]-np.mean(b[0,0:length])

# Use butter filter
if True:
    BPM_lowest = 40
    BPM_max = 230
    Wn = [((BPM_lowest/60)/fps * 2), ((BPM_max/60)/fps * 2)]
    B_butter, A_butter = butter(N=2, Wn=Wn, btype='band')
    r = filtfilt(B_butter, A_butter, r)
    g = filtfilt(B_butter, A_butter, g)
    b = filtfilt(B_butter, A_butter, b)

fps = 30
print("GREEN")
tests.interval_test(g,fps)
print("BLUE")
tests.interval_test(b,fps)
