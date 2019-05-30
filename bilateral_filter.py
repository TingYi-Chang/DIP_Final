import numpy as np
import cv2
import sys
import math

def bilateral_filter(image):
	#image = cv2.imread(sys.argv[1])

	#cv2.imshow("Original", image)
	#blurred = np.hstack([
		#cv2.bilateralFilter(image, 5, 21, 21),
		#cv2.bilateralFilter(image, 7, 31, 31),
		#cv2.bilateralFilter(image, 9, 75, 75)])
	#cv2.namedWindow( "Bilateral", cv2.WINDOW_NORMAL )
	#cv2.resizeWindow("Bilateral", 1000, 800)
	blurred = cv2.bilateralFilter(image)
	#cv2.imshow("Bilateral", blurred)
	#cv2.waitKey(0)
	return blurred
