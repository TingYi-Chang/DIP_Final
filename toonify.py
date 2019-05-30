import cv2
import numpy as np 
import math
import sys
from matplotlib import pyplot as plt


def toonify_DOG_bilateral()






def toonify_canny_bilateral()











if len(sys.argv) == 2:
    filename = (sys.argv[1])
else:
    print ("ERROR INPUT ")
    exit()

input_image = cv2.imread(filename)
result = cv2.blur(input_image, (5,5))
 
cv2.imshow("Origin", input_image)
cv2.imshow("Blur", result)
cv2.waitKey(0)

