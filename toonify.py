import cv2
import numpy as np 
import math
import sys
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans




def Quantize_Colors(image, n_clusters):
	(h, w) = image.shape[:2]
	image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
 
	# reshape the image into a feature vector so that k-means
	# can be applied
	image = image.reshape((image.shape[0] * image.shape[1], 3))
	 
	# apply k-means using the specified number of clusters and
	# then create the quantized image based on the predictions
	clt = MiniBatchKMeans(n_clusters = n_clusters )
	labels = clt.fit_predict(image)
	quant = clt.cluster_centers_.astype("uint8")[labels]
	 
	# reshape the feature vectors to images
	quant = quant.reshape((h, w, 3))
	image = image.reshape((h, w, 3))
	 
	# convert from L*a*b* to RGB
	quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
	image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
	 
	# display the images and wait for a keypress
	#cv2.imshow("image", np.hstack([image, quant]))
	return quant


def toonify_DOG_bilateral(input_image, filename, gray_image):
	# step1: edge detection use DOG #

	#gray_image = cv2.medianBlur(gray_image, 7),

	

	outputFile = str(filename)[:-4] + '_DoG.jpg'
	
	#edge = cv2.Canny(input_image,1500,250)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4, 4))
	#edge = cv2.dilate(edge,kernel)

	blur5 = cv2.GaussianBlur(gray_image,(13,13),0)
	blur3 = cv2.GaussianBlur(blur5,(9,9),0)
	edge = blur5 - blur3
	# edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel)
	# kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5, 5))
	# edge = cv2.dilate(edge,kernel)

	edge = cv2.Canny(input_image,170,200)
	cv2.imwrite("dog.jpg", edge)

	# step2: REGION SMOOTHENING use bilateral filter #

	smooth = cv2.bilateralFilter(input_image, 20, 50, 50)
	smooth = Quantize_Colors(smooth, 15)
	#cv2.imwrite(outputFile, smooth)

	# step3: combine #
	d = 0
	for i in range(edge.shape[0]):
		for j in range(edge.shape[1]):
			if edge[i][j] == 255:
				d += 1
				smooth[i][j][0] = 0
				smooth[i][j][1] = 0 
				smooth[i][j][2] = 0
	#print(edge)
	cv2.imwrite(outputFile, smooth)


#def toonify_canny_bilateral()











if len(sys.argv) == 2:
	filename = (sys.argv[1])
else:
	print ("ERROR INPUT ")
	exit()

input_image = cv2.imread(filename, cv2.IMREAD_COLOR)
input_image_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
#result = cv2.blur(input_image, (5,5))

toonify_DOG_bilateral(input_image, filename, input_image_gray)
#cv2.imshow("Origin", input_image)
#cv2.imshow("Blur", result)
#cv2.waitKey(0)

