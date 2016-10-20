import cv2
import numpy as np

def auto_canny(image, sigma=0.33):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
 
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)

	return edged
 
	# return the edged image
	return edged
x = cv2.imread('data/out0001.png-best.png')
x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
x = cv2.medianBlur(x, 3)
x = 255-auto_canny(x)
cv2.imwrite('test1.png', x)
x = cv2.distanceTransform(x, cv2.cv.CV_DIST_L2, 3)
cv2.imwrite('test2.png', x)