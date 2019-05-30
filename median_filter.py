import numpy as np
import cv2
import sys
import math

def median_filter(image):
	#image = cv2.imread(sys.argv[1])
	median = cv2.medianBlur(image,5)
	#cv2.imshow("Median",median)
	#cv2.waitKey(0)
	return median
