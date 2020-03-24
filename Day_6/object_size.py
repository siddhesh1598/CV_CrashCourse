# pixels_per_metric = object_width / know_width

# import
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import cv2
import imutils
import argparse

def midpoint(ptA, ptB):
	return((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

# construct argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width(in inchs) of the left-most object")
args = vars(ap.parse_args())

# load image, convert to rgayscale and blur
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# perform edge detection and perform 
# dilation + erosion to close gaps between object edges
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)

# find contours in the edged map
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# sort the contours from left-to-right
(cnts, _ ) = contours.sort_contours(cnts)
pixelsPerMetric = None

# loop over individual contours
for c in cnts:
	# ignore small contours
	if cv2.contourArea(c) < 100:
		continue

	# compute the rotated bounding box of the contour
	orig = image.copy()
	box = cv2.minAreaRect(c)
	box = cv2.boxPoints(box) 
	box = np.array(box, dtype="int")

	# order the points in the contour such that 
	# they appear in top-left, top-right, bottom-right, 
	# and bottom-left order
	# draw the outline of the rotated bounding box
	box = perspective.order_points(box)
	cv2.drawContours(orig, [box.astype("int")], 
		-1, (0, 255, 0), 2)

	# loop over original points and draw them
	for (x, y) in box:
		cv2.circle(orig, (int(x), int(y)), 5, 
			(0, 0, 255), -1)

	# unpack the co-ordinates of the box and
	# calculate their midboints
	(tl, tr, br, bl) = box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)

	# compute midpoint between the tl and bl points and
	# the midpoint between the tr and br
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# draw the midpoints on the image
	cv2.circle(orig, (int(tltrX), int(tltrY)), 
		5, (255, 0, 0), -1)
	cv2.circle(orig, (int(blbrX), int(blbrY)), 
		5, (255, 0, 0), -1)
	cv2.circle(orig, (int(tlblX), int(tlblY)), 
		5, (255, 0, 0), -1)
	cv2.circle(orig, (int(trbrX), int(trbrY)), 
		5, (255, 0, 0), -1)
	
	# draw lines between the midpoints
	cv2.line(orig, (int(tltrX), int(tltrY)), 
		(int(blbrX), int(blbrY)),
		(255, 0, 255), 2)
	cv2.line(orig, (int(tlblX), int(tlblY)), 
		(int(trbrX), int(trbrY)),
		(255, 0, 255), 2)

	# Compute the Euclidean dist between the midpoints
	# height
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	# width
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# compute the pixels per metric
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]

	# compute the size of the object
	dimA = dA / pixelsPerMetric
	dimB = dB / pixelsPerMetric

	# draw object sizes on the images
	cv2.putText(orig, "{:.1f}in".format(dimA),
		(int(tltrX-15), int(tltrY-10)), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, 
		(255, 255, 255), 2)
	cv2.putText(orig, "{:.1f}in".format(dimB),
		(int(trbrX + 10), int(trbrY)), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.65,
		(255, 255, 255), 2)

	# show the image
	cv2.imshow("Image", orig)
	cv2.waitKey(0)
