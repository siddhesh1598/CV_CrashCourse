# import 
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", 
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int,
	default=64, help="max buffer size")
args = vars(ap.parse_args())

# define lower and upper bound of green ball
greenLower = (29, 86, 6)
greenUpper = (64, 255, 255)

# initialize list of traked points
pts = deque(maxlen=args["buffer"])

# select video path or webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
else:
	vs = cv2.VideoCapture(args["video"])

# allow the camera or video file to warm up
time.sleep(2.0)

# keep looping
while True:
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame

	if frame is None:
		break

	# resize the frame, blur it and convert to HSV color
	frame = imutils.resize(frame, width=600)
	blurred = cv2.GaussianBlur(frame, (11, 11), 0)
	hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

	# construct a mask of green and then perform
	# series of dilation and erosion to remove
	# any small blobs left in the mask
	mask = cv2.inRange(hsv, greenLower, greenUpper)
	mask = cv2.erode(mask, None, iterations=2)
	mask = cv2.dilate(mask, None, iterations=2)

	# find contours in the mask and initialize 
	# the center of the ball
	cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	center = None

	# only proceed if atleat one contour is found
	if len(cnts) > 0:
		# find max contour and draw circle around it
		c = max(cnts, key=cv2.contourArea)
		((x, y), radius) = cv2.minEnclosingCircle(c)
		M = cv2.moments(c)
		center = (int(M["m10"] / M["m00"]), 
				int(M["m01"] / M["m00"]))

		# only proceed if the radius meets the minimum size
		if radius > 10:
			# draw a circle and centroid on the frame
			cv2.circle(frame, (int(x), int(y)), int(radius),
				(0, 255, 255), 2)
			cv2.circle(frame, center, 5, (0, 0, 255), -1)

	# update the list of tracking points
	pts.appendleft(center)

	# loop over the set of tracked points
	for i in range(1, len(pts)):
		if pts[i-1] is None or pts[i] is None:
			continue

		# compute thickness of the line 
		thickness = int(np.sqrt(args["buffer"] / float(i+1)) * 2.5)
		# draw the connecting line
		cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), 
					thickness)

	# show the frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

# stop webcam if in use
if not args.get("video", False):
	vs.stop()
# otherwise release the camera
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()

