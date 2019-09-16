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


def toonify_DOG_bilateral(QN, input_image, filename, gray_image):
	# step1: edge detection use Canny #
	input_image = cv2.medianBlur(input_image, 3)

	edge = cv2.Canny(input_image,180,240)
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2, 2))
	edge = cv2.dilate(edge,kernel)


	cv2.imwrite("edge_C.jpg", edge)

	# step2: REGION SMOOTHENING use bilateral filter #
	
	smooth = cv2.bilateralFilter(input_image, 20, 50, 50)
	for i in range(5):
		smooth = cv2.bilateralFilter(smooth, 20, 50, 50)
	smooth = Quantize_Colors(smooth, QN)
	
	# step3: combine #
	
	for i in range(edge.shape[0]):
		for j in range(edge.shape[1]):
			if edge[i][j] == 255:
				smooth[i][j][0] /= 2
				smooth[i][j][1] /= 2 
				smooth[i][j][2] /= 2
	outputFile = str(filename)[:-4] + '_toonified.jpg'
	cv2.imwrite(outputFile, smooth)
	






if len(sys.argv) >= 2:
	filename = (sys.argv[2])
else:
	print ("ERROR INPUT ")
	exit()
if (sys.argv[1] == "-m"):
	QN = 5;
elif (sys.argv[1] == "-p"): 
	QN = 10;
input_image = cv2.imread(filename, cv2.IMREAD_COLOR)
input_image_gray = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

toonify_DOG_bilateral(QN, input_image, filename, input_image_gray)

