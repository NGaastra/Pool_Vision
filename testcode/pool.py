import cv2
import numpy as np

ball1Lower, ball1Upper = (0, 190, 250), (0, 198, 255)
tableLower, tableUpper = (1, 43, 0), (6, 112, 13)

img = cv2.imread('pooltable.png', 1)

mask = cv2.inRange(img, tableLower, tableUpper)
mask = cv2.dilate(mask, None, iterations=1)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True
params.minArea = 1
params.maxArea = 1000
params.filterByColor = True
params.blobColor = 0
params.filterByCircularity = True
params.minCircularity = 0.1
params.filterByInertia = False
params.filterByConvexity = False

# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs.
keypoints = detector.detect(mask)

im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (255,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow('blobs', im_with_keypoints)
cv2.imshow('mask', mask)

'''
mask = cv2.inRange(img, tableLower, tableUpper)
mask = cv2.dilate(mask, None, iterations=1)



edged = cv2.Canny(mask, 30, 200)
edged = cv2.dilate(edged, None, iterations=1)

circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, 1, 100)

# ensure at least some circles were found
if circles is not None:
    print('222')
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

cv2.imshow('edged', edged)
cv2.imshow('mask', mask)
cv2.imshow('img', img)
#gray = cv2.bilateralFilter(gray, 11, 17, 17)
#edged = cv2.Canny(img, 30, 200)
# detect circles in the image

circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 1.2, 100)

# ensure at least some circles were found
if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(img, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(img, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

cv2.imshow('frame', edged)
cv2.imshow('gray', img)
'''
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
while 1:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

img.release()
cv2.destroyAllWindows()