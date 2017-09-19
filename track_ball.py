import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

# Load image in BGR (params: image_path)
# params: image_path
# return: inputs derived image object
from pip.utils import dist_in_site_packages


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
# params: image, search_range = 45, shadow_intensity = 20, blur = True
# return: guessed table mask
def getTableMask(image, search_range = 45, shadow_intensity = 20, blur = True):
    # Get most common color in 'image' with a certain color range
    min_color, max_color = getTableColor(image, search_range, shadow_intensity)

    #Get mask that fits the previous gotten color range
    mask = cv2.inRange(image, min_color, max_color)
    # Blur mask
    if blur:
        mask = cv2.dilate(mask, None, iterations=1)
        mask = cv2.bilateralFilter(mask, 3, 175, 175)

    return mask

def getBallMask(crop, bg):
    centers = subBackground(bg, crop)

    #centers = cv2.bitwise_and(crop, crop, mask = new_mask)
    #centers = cv2.cvtColor(centers, cv2.COLOR_BGR2GRAY)
    _, contours, _ = cv2.findContours(centers, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    '''
    # Get distance transform from table mask
    dist = cv2.distanceTransform(new_mask, cv2.DIST_L2, 5);

    # Threshold distance transform values by 1/4 max value
    _, centers = cv2.threshold(dist, 0.25 * dist.max(), 255, cv2.THRESH_BINARY)

    # Dilate to expand found circles
    centers = cv2.dilate(centers, None, iterations=5)
    '''
    return centers

# Get table color
# params: image, search_range
# return: guessed minimum color and maximum color of inputted image
def getTableColor(hsv, search_range = 25, shadow_intensity = 0):
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
# return: array with 4 coordinates which correspond to table corners in this order: bottom left (x, y), top left (x, y), top right (x, y), bottom right (x, y)
def getTableCorners(table_contour):
	# Get min area rectangle from 'table_contour', which corrects for rotation
	box = cv2.boxPoints(cv2.minAreaRect(table_contour))
	return [[box[0][0], box[0][1]], [box[1][0], box[1][1]], [box[2][0], box[2][1]], [box[3][0], box[3][1]]]

# Get all balls from table
# params: table_mask
# return: gets all balls that can be detected in 'table_mask'
def getBallsContour(mask, img):
    mask = np.array(mask, np.uint8)
    delete = []
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, cnt in enumerate(contours):
        _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for i, cnt in enumerate(contours):
            x, y, w, h = cv2.boundingRect(cnt)
            if not (0.95 < w / h < 1.05):
                delete.append(i)
    contours  = np.delete(contours , delete)
    return contours
# Crop image according to table
# params: image, table_contour
# return: image cropped to 'table_contour' size
def	isolateTable(image, table_contour):
	corners = getTableCorners(table_contour) 
	print([int(corners[1][1]),int(corners[3][1]), int(corners[1][0]),int(corners[3][0])])
	# Crop by slicing array in this order: start y, end y, start x, end x
	cropped = image[int(corners[1][1]):int(corners[3][1]), int(corners[1][0]):int(corners[3][0])]
	cropped = imutils.resize(cropped, height=500)
	return cropped

def drawBalls(balls, img):
    for cnt in balls:
        M = cv2.moments(cnt)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cv2.circle(img, (cX, cY), 15, (0, 255, 0), 2)

def subBackground(bg, image):
    # Get table mask and bg mask
    table_mask = cv2.bitwise_not(getTableMask(toHSV(image)))
    bg_mask = cv2.bitwise_not(getTableMask(toHSV(bg)))

    # Subtract background from original image
    return table_mask - bg_mask


def initTable(image):
    img = loadImage(image)
    bg = loadImage('img\pooltableempty.png')
    hsv = toHSV(img)
    mask = getTableMask(hsv)
    table_contour = getTableContour(mask)
    cropped = isolateTable(img, table_contour)
    cropped_bg = isolateTable(bg, table_contour)
    ball_mask = getBallMask(cropped, cropped_bg)
    #balls = getBallsContour(ball_mask, cropped)
    #drawBalls(balls, cropped)
    #debug
    cv2.imshow('image', cropped)
    cv2.imshow('mask', mask)
    cv2.imshow('balls', ball_mask)
	
initTable('img\pooltable.png')


while 1:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()