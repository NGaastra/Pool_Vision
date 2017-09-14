"""
Created on Sun Jul 12 21:32:26 2015
@author: Stuart Grieve
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import Indexer


def Test():
    img = LoadImage('pooltable.png')

    hsv = ToHSV(img)

    lower_color, upper_color = GetClothColor(hsv)

    contours = GetContours(hsv, lower_color, upper_color, 7)

    TableContour = MaskTableBed(contours)
    cv2.imshow('contour', hsv)

def CueBall(BallData):
    data = BallData[1][2]

    # this mask does not reflect the boundary between data and nodata.
    mask = cv2.inRange(data, (0, 0, 10), (180, 255, 255))

    #    cv2.imshow('result1',mask)
    #    cv2.imshow('result',data)
    #
    #    cv2.waitKey(0)
    #    cv2.destroyAllWindows()


    hist = cv2.calcHist([data], [0], mask, [180], [0, 180])

    plt.plot(hist)
    plt.show()

    hist = cv2.calcHist([data], [1], mask, [256], [0, 256])

    plt.plot(hist)
    plt.show()

    hist = cv2.calcHist([data], [2], mask, [256], [0, 256])

    plt.plot(hist)
    plt.show()


def LoadImage(filename):
    """
    Loads an image file
    """
    # img is loaded in bgr colorspace
    return cv2.imread(filename)


def ToHSV(img):
    """
    Convert an image from BGR to HSV colorspace
    """
    return cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)


def GetClothColor(hsv, search_width=45):
    """
    Find the most common HSV values in the image.
    In a well lit image, this will be the cloth
    """

    hist = cv2.calcHist([hsv], [1], None, [180], [0, 180])
    h_max = Indexer.get_index_of_max(hist)[0]

    hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
    s_max = Indexer.get_index_of_max(hist)[0]

    hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
    v_max = Indexer.get_index_of_max(hist)[0]

    # define range of blue color in HSV
    lower_color = np.array([h_max - search_width, s_max - search_width, v_max - search_width])
    upper_color = np.array([h_max + search_width, s_max + search_width, v_max + search_width])

    return lower_color, upper_color


def MaskTableBed(contours):
    """
    Mask out the table bed, assuming that it will be the biggest contour.
    """

    # The largest area should be the table bed
    areas = []
    for c in contours:
        areas.append(cv2.contourArea(c))

    # return the contour that delineates the table bed
    largest_contour = Indexer.get_index_of_max(areas)
    return contours[largest_contour[0]]


def GetContours(hsv, lower_color, upper_color, filter_radius):
    """
    Returns the contours generated from the given color range
    """

    # Threshold the HSV image to get only cloth colors
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # use a median filter to get rid of speckle noise
    median = cv2.medianBlur(mask, filter_radius)

    # get the contours of the filtered mask
    # this modifies median in place!
    _, contours, _ = cv2.findContours(median, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    return contours

Test()

