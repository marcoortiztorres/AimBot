
# USAGE
# python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel

# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
from collections import OrderedDict
import numpy as np
import argparse
import imutils
import time
import cv2
import serial
from time import sleep

#setting up Bluetooth connection
bluetoothSerial = serial.Serial("/dev/rfcomm1", baudrate=9600)
#bt2 = serial.Serial("/dev/rfcomm0", baudrate=9600)

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()

time.sleep(0.5)
fps = FPS().start()

# while there is live video loop thorugh each frame
while True:
	frame = vs.read()
	frame = imutils.resize(frame, width=200)

	commandX = ""
	commandY = ""
	commandLED = 'G'
	xCentered = False
	yCentered = False

	# for copy pasta code
	image = frame

	blurred = cv2.GaussianBlur(image, (5, 5), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	lower_red_1 = np.array([0,70,50])
	upper_red_1 = np.array([10,255,255])

	lower_red_2 = np.array([170,70,50])
	upper_red_2 = np.array([180,255,255])
	red_mask1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
	red_mask_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

	red_mask = red_mask1 | red_mask_2

	_, red_contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


	itemFound = False
	for contour in red_contours:
		area = cv2.contourArea(contour)

		if area > 1000:
			M = cv2.moments(contour)
			if M["m00"] == 0.0:
				M["m00"] = 0.001
			cX = int(M["m10"] / M["m00"])
			cY = int(M["m01"] / M["m00"])
			#cv2.drawContours(image, contour, -1, (0, 0, 255), 3)
			#cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
			#cv2.putText(image, str(cX)+","+str(cY), (cX - 20, cY - 20),
			#	cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			#print("X: ", cX, " Y: ", cY)

			if (cX < 80):
				commandX = 'D' #'R'
			elif (cX > 120):
				commandX = 'U' #'L'
			else: xCentered = True
			if (cY < 80):
				commandY = 'R' #'U'
			elif (cY > 120):
				commandY = 'L' #'D'
			else:

                            yCentered = True
			itemFound = True

			break
	if not itemFound:
		commandY = 'L'
	if (xCentered and yCentered and itemFound):
		commandLED = 'K'
	#res = cv2.bitwise_and(frame, frame, mask=red_mask)
	#res = cv2.medianBlur(res, 5)

	# cv2.imshow('Original image', frame)
	#cv2.imshow('Color Detector', res)
	#cv2.imshow('mask',red_mask)
	if (commandX != ""):
		bluetoothSerial.write(commandX.encode())
		#bluetoothSerial.write(str("X").encode())
	if (commandY != ""):
		bluetoothSerial.write(commandY.encode())
		#bluetoothSerial.write(str("X").encode())
	time.sleep(0.1)
	bluetoothSerial.write(str("X").encode())
	bluetoothSerial.write(commandLED.encode())
	#print("\nCommandX: ", commandX, "\nCommand Y: ", commandY)
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
