import cv2
import numpy as np

ball1Lower, ball1Upper = (0, 190, 250), (0, 198, 255)
tableLower, tableUpper = (1, 43, 0), (6, 112, 13)

img = cv2.imread('pooltable.png', 1)

orb = cv2.ORB_create(1)

kp, des = orb.detectAndCompute(img,None)

print(len(kp))
img = cv2.drawKeypoints(img, kp, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('mask', img)

while 1:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

img.release()
cv2.destroyAllWindows()