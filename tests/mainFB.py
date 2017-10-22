import numpy as np
import tests
import argparse
import cv2

parser = argparse.ArgumentParser(description="Photoplesthysmography")
parser.add_argument("--file", "-f", type=str, dest='file', default='/home/francis/Desktop/videodedo.mp4')

args = parser.parse_args()

cap = cv2.VideoCapture(args.file)

if not cap.isOpened():
   print("No lo pude abrir")

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

r = np.zeros((1, length))
g = np.zeros((1, length))
b = np.zeros((1, length))

k = 0
while (cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        r[0, k] = np.mean(frame[330:360, 610:640, 0])
        g[0, k] = np.mean(frame[330:360, 610:640, 1])
        b[0, k] = np.mean(frame[330:360, 610:640, 2])
    # print(k)
    else:
        break
    k = k + 1

cap.release()
cv2.destroyAllWindows()

r = r[0,0:length]-np.mean(r[0,0:length])
g = g[0,0:length]-np.mean(g[0,0:length])
b = b[0,0:length]-np.mean(b[0,0:length])

tests.interval_test(g)


