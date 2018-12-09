# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from collections import OrderedDict
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import time
import cv2

class ColorLabeler:
	def __init__(self):
		#Create color dictionary
		colors = OrderedDict({"green": (0,255,0), "red": (255,0,0), "blue": (0,0,255)})

		# allocate memory for the L*a*b* image, then initialize
		# the color names list
		self.lab = np.zeros((len(colors), 1, 3), dtype="uint8")
		self.colorNames = []

		# loop over the colors dictionary
		for (i, (name, rgb)) in enumerate(colors.items()):
			# update the L*a*b* array and the color names list
			self.lab[i] = rgb
			self.colorNames.append(name)

		# convert the L*a*b* array from the RGB color space
		# to L*a*b*
		self.lab = cv2.cvtColor(self.lab, cv2.COLOR_RGB2LAB)

	def label(self, image, c):
		# construct a mask for the contour, then compute the
		# average L*a*b* value for the masked region
		mask = np.zeros(image.shape[:2], dtype="uint8")
		cv2.drawContours(mask, [c], -1, 255, -1)
		mask = cv2.erode(mask, None, iterations=2)
		mean = cv2.mean(image, mask=mask)[:3]

		# initialize the minimum distance found thus far
		minDist = (np.inf, None)

		# loop over the known L*a*b* color values
		for (i, row) in enumerate(self.lab):
			# compute the distance between the current L*a*b*
			# color value and the mean of the image
			d = dist.euclidean(row[0], mean)

			# if the distance is smaller than the current distance,
			# then update the bookkeeping variable
			if d < minDist[0]:
				minDist = (d, i)

		# return the name of the color with the smallest distance
		return self.colorNames[minDist[1]]


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)
fps = FPS().start()

# while there is live video loop thorugh each frame
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# for copy pasta code
	image = frame

	blurred = cv2.GaussianBlur(image, (5, 5), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	lower_green = np.array([40, 20, 20])
	upper_green = np.array([70,255,255])

	lower_red_1 = np.array([0,70,50])
	upper_red_1 = np.array([10,255,255])

	lower_red_2 = np.array([170,70,50])
	upper_red_2 = np.array([180,255,255])

	lower_blue = np.array([25, 50, 50])
	upper_blue = np.array([32,255,255])

	green_mask = cv2.inRange(hsv, lower_green, upper_green)
	_,green_contours,_ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	red_mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
	red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

	red_mask = red_mask1 | red_mask_2

	_, red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
	_, blue_contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

	# cv2.drawContours(image, green_contours,-1,cvScalar(0, 255, 0),3)
	# cv2.drawContours(image, red_contours, -1, cvScalar(255, 0, 0), 3)

	for contour in green_contours:
		area = cv2.contourArea(contour)
		if area > 1000:
			cv2.drawContours(image, contour, -1, (0, 255, 0), 3)
			#print(area)

	for contour in red_contours:
		area = cv2.contourArea(contour)

		# compute center of contour
		M = cv2.moments(contour)
		if M["m00"] == 0.0:
			M["m00"] = 0.001
		cX = int(M["m10"] / M["m00"])
		cY = int(M["m01"] / M["m00"])

		if area > 1000:
			cv2.drawContours(image, contour, -1, (0, 0, 255), 3)
			cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
			cv2.putText(image, str(cX)+","+str(cY), (cX - 20, cY - 20),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			#print(area)

	for contour in blue_contours:
		area = cv2.contourArea(contour)
		if area > 1000:
			cv2.drawContours(image, contour, -1, (255, 0, 0), 3)
			#print(area)

	all_masks = green_mask | red_mask | blue_mask

	res = cv2.bitwise_and(frame, frame, mask=all_masks)
	res = cv2.medianBlur(res, 5)

	# cv2.imshow('Original image', frame)
	cv2.imshow('Color Detector', frame)
	# cv2.imshow('mask',all_masks)

	# Check if the user pressed ESC key
	c = cv2.waitKey(1)
	if c == 27:
		break
		# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
