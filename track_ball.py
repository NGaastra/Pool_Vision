import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load image in BGR (params: image_path)
# params: image_path
# return: inputs derived image object
def loadImage(image_path):
    return cv2.imread(image_path, 1)

# Convert BGR image to HSV image 
# params: image
# reutrn: input image's HSV version
def toHSV(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Get distance between two points 
# params: x1, y1, x2, y2
# return: distance between two coordinate points given
def getDist(x1, y1, x2, y2):
    return np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))

# Get table mask
# params: image, search_range = 45, shadow_intensity = 20, blur = 3
# return: table mask with size 'blur' median blur
def getTableMask(image, search_range = 45, shadow_intensity = 20, blur = 3):
    # Get most common color in 'image' with a certain color range
	min_color, max_color = getTableColor(image, search_range, shadow_intensity)
	
	#Get mask that fits the previous gotten color range
	mask = cv2.inRange(image, min_color, max_color)
	
	# Blur mask with size 'blur' median blur
	clean_mask = cv2.medianBlur(mask, blur)
	
	return clean_mask
	
# Get table color
# params: image, search_range
# return: guessed minimum color and maximum color of inputted image
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

# Get table contour
# params: table_mask
# return: biggest contour detected (which is the table)
def getTableContour(table_mask):
	# Get all contours in given mask
	_, contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	
	#Sort contours by area from high to low and get first entry
	table_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
	return table_contour
	
# Get table corners
# params: table_contour
# return: array with 4 coordinates which correspond to table corners in this order: bottom left, top left, top right, bottom right
def getTableCorners(table_contour):
	# Get min area rectangle from 'table_contour', which corrects for rotation
	box = cv2.boxPoints(cv2.minAreaRect(table_contour))
	return [[box[0][0], box[0][1]], [box[1][0], box[1][1]], [box[2][0], box[2][1]], [box[3][0], box[3][1]]]

img = loadImage('pooltable.png')
hsv = toHSV(img)
mask = getTableMask(hsv, blur=7)
table_contour = getTableContour(mask)
corn = getTableCorners(table_contour)

cv2.imshow('image', img)
cv2.imshow('mask', mask)

while 1:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()