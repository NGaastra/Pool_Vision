from re import sub

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

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


class Table:

    def __init__(self, table_bg):
        self.table_bg = loadImage(table_bg)
        self.table_color_min, self.table_color_max = self.getTableColor(toHSV(self.table_bg))
        self.table_mask = self.getTableMask(toHSV(self.table_bg))
        self.table_contour = self.getTableContour(self.table_mask)
        self.table_cropped = self.isolateTable(self.table_bg, self.table_contour)

        self.table_cur = loadImage('img\pooltable.png')
        self.table_cur_mask = self.getTableMask(toHSV(self.table_cur))

    # Get table mask
    # params: image, search_range = 45, shadow_intensity = 20, blur = True
    # return: guessed table mask
    def getTableMask(self, image, blur = True):
        # Get most common color in 'image' with a certain color range

        #Get mask that fits the previous gotten color range
        mask = cv2.inRange(image, self.table_color_min, self.table_color_max)
        # Blur mask
        if blur:
            mask = cv2.dilate(mask, None, iterations=1)
            mask = cv2.bilateralFilter(mask, 3, 175, 175)

        return mask

    def getBallMask(self, image):
        # Subtract background
        ball_mask = self.subBackground()

        # Remove all small noise specks from the mask
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, None)

        #Combine the mask with the current image
        masked_image = cv2.bitwise_and(image, image, mask = ball_mask)

        # Return the mask and the image
        return ball_mask, masked_image

    # Get table color
    # params: image, search_range
    # return: guessed minimum color and maximum color of inputted image
    def getTableColor(self, hsv, search_range = 45, shadow_intensity = 20):
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
    def getTableContour(self, table_mask):
        # Get all contours in given mask
        _, contours, _ = cv2.findContours(table_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #Sort contours by area from high to low and get first entry
        table_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
        return table_contour

    # Get table corners
    # params: table_contour
    # return: array with 4 coordinates which correspond to table corners in this order: bottom left (x, y), top left (x, y), top right (x, y), bottom right (x, y)
    def getTableCorners(self, table_contour):
        # Get min area rectangle from 'table_contour', which corrects for rotation
        box = cv2.boxPoints(cv2.minAreaRect(table_contour))
        return [[box[0][0], box[0][1]], [box[1][0], box[1][1]], [box[2][0], box[2][1]], [box[3][0], box[3][1]]]

    # Get all balls from table
    # params: table_mask
    # return: gets all balls that can be detected in 'table_mask'
    def getBallsContour(self, ball_mask):
        _, contours, _ = cv2.findContours(ball_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    # Crop image according to table
    # params: image, table_contour
    # return: image cropped to 'table_contour' size
    def	isolateTable(self, image, table_contour):
        corners = self.getTableCorners(table_contour)
        print([int(corners[1][1]),int(corners[3][1]), int(corners[1][0]),int(corners[3][0])])
        # Crop by slicing array in this order: start y, end y, start x, end x
        cropped = image[int(corners[1][1]):int(corners[3][1]), int(corners[1][0]):int(corners[3][0])]
        cropped = imutils.resize(cropped, width=500)
        return cropped

    def subBackground(self):
        bg_mask = cv2.bitwise_not(self.table_mask)
        table_mask = cv2.bitwise_not(self.table_cur_mask)
        return table_mask - bg_mask

    def drawMean(self):
        _, contours, _ = cv2.findContours(self.table_cur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        final = np.zeros(self.table_cur.shape, np.uint8)
        mask = np.zeros(self.table_cur.shape, np.uint8)

        for i in range(len(contours)):
            mask[...] = 0
            cv2.drawContours(mask, contours, i, 255, -1)
            cv2.drawContours(final, contours, i, cv2.mean(self.table_cur, mask), -1)
            print(i, '\t', cv2.mean(self.table_cur, mask))

        return final

def initTable(table_bg):
    table = Table(table_bg)
    ball_mask, test_mask = table.getBallMask(loadImage('img\pooltable.png'))
    table.drawMean()
    #debug
    cv2.imshow('bg', table.table_mask)
    cv2.imshow('bm', ball_mask)
    cv2.imshow('tm', test_mask)

initTable('img\pooltableempty.png')


while 1:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()