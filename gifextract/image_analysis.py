import cv2
import numpy as np
import glob

imgpaths = glob.glob('png/*.png')

for filename in sorted(imgpaths, key=lambda x: int(x.split('-')[1][:-4])):
    img = cv2.imread(filename)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,np.array([0,90,0]),np.array([100,255,255]))
    attempt = cv2.bitwise_and(img, img, mask = mask)

    mask2 = cv2.inRange(attempt,np.array([20,40,40]),np.array([30,50,50]))
    attempt2 = cv2.bitwise_and(attempt, attempt, mask = cv2.bitwise_not(mask2))

    outname = filename.split('/')[1]
    cv2.imwrite('out/{}'.format(outname), attempt2)
