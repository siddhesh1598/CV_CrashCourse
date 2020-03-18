# import 
import imutils
import cv2

# read image
image = cv2.imread("jp.png")

# get dimensions of the image
(h, w, d) = image.shape
print("Width: {}, Height: {}, Depth: {}".format(w, h, d))

# display the image
cv2.imshow("Image", image)
cv2.waitKey(0)

# getting pixel value
# height = 100	width = 50
(B, G, R) = image[100, 50]
print("Red: {}, Green: {}, Blue: {}".format(R, G, B))

# array slicing to extract ROI
# image[startY:endY, startX:endX]
roi = image[60:160, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)

# resize the image to 200x200
resized = cv2.resize(image, (200, 200))
cv2.imshow("Fixed Resizing", resized)
cv2.waitKey(0)

# resize image to width = 300 
# maintianing the aspect ratio
r = 300.0 / w	# ratio = new / old
dim = (300, int(h * r))
resized = cv2.resize(image, dim)
cv2.imshow("Aspect Ratio Resize", resized)
cv2.waitKey(0)

# maintaining the aspect ratio using imutils
resized = imutils.resize(image, width=300)
cv2.imshow("Imutils Resize", resized)
cv2.waitKey(0)

# rotate the image 45 degrees clock-wise
center = (w // 2, h // 2)
# create a matrix with center, degree and scale to zoom
M = cv2.getRotationMatrix2D(center, -45, 2.0)
# actually rotate the image
rotated = cv2.warpAffine(image, M, (w,h))
cv2.imshow("OpenCV Rotation", rotated)
cv2.waitKey(0)

# performing rotation using imutils
rotated = imutils.rotate(image, -45)
cv2.imshow("Imutils Rotation", rotated)
cv2.waitKey(0)

# rotate while avoiding image clipping
# the degree signs are vice-versa 
# i.e. +ve for clock-wise rotation
rotated = imutils.rotate_bound(image, 45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

# applying Gaussian Blur to reduce noise 
# and to smoothen the image
# Larger kernels -> more blurry image
# Smaller kernels -> less blurry images
# cv2.GaussianBlur(image, kernel size, sigma)
blurred = cv2.GaussianBlur(image, (11, 11), 0)
cv2.imshow("Blurred", blurred)
cv2.waitKey(0)

# draing on the image
output = image.copy()
# cv2.rectangle(image, top-left, bottom-right, 
#			color, thickness)
cv2.rectangle(output, (320, 60), (420, 160), 
			(0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

output = image.copy()
# cv2.circle(image, center, radius,
#			color, thickness)
cv2.circle(output, (300, 150), 20, 
			(255, 0, 0), -1)
cv2.imshow("Circle", output)
cv2.waitKey(0)

output = image.copy()
# cv2.line(image, start, end, 
#			color, thickness)
cv2.line(output, (60, 20), (400, 200), 
			(0, 0, 255), 5)
cv2.imshow("Line", output)
cv2.waitKey(0)

# display text over the image
output = image.copy()
# cv2.putText(image, text, 
#			start-point, font,
#			size, color, thickness)
cv2.putText(output, "OpenCV + Jurassic Park!!!",
			(10, 25), cv2.FONT_HERSHEY_SIMPLEX, 
			0.7, (0, 255, 0), 2)
cv2.imshow("Text", output)
cv2.waitKey(0)
