import cv2
import numpy as np
from matplotlib import pyplot as plt

import Indexer

# Load image in BGR (params: image_path)
def loadImage(image_path):
    return cv2.imread(image_path, 1)

# Convert BGR image to HSV image (params: image)
def toHSV(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Get distance between two points (params: x1, y1, x2, y2)
def getDist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

# Tries to mask table in HSV image (params: image, search_range = 45, shadow_intensity = 20)
def getTableMask(image, search_range = 45, shadow_intensity = 20):
    # Get most common color in 'image' with a certain range
    min_color, max_color = getTableColor(image, search_range, shadow_intensity)
    return cv2.inRange(image, min_color, max_color)

# Guess table color with high succes rate (params: image, search_range)
def getTableColor(hsv, search_range, shadow_intensity):
    # Get histogram from image's hue channel and get max value
    hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    h_max = np.argmax(hist)

    # Get histogram from image's saturation channel and get max value
    hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    s_max = np.argmax(hist)

    # Get histogram from image's value channel and get max value
    hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    v_max = np.argmax(hist)

    # Define range of color in HSV and make lower_color's brightness lower depending on shadow interference
    lower_color = np.array([h_max - search_range, s_max - search_range, v_max - search_range - shadow_intensity])
    upper_color = np.array([h_max + search_range, s_max + search_range, v_max + search_range])

    return lower_color, upper_color

img = loadImage('pooltable3.jpg')
hsv = toHSV(img)
mask = getTableMask(hsv)

cv2.imshow('image', img)
cv2.imshow('mask', mask)

while 1:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()