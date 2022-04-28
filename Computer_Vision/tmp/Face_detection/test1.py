import cv2
import numpy
def detect(img):
	cascade = cv2.CascadeClassifier("cascade.xml")
	rects = cascade.detectMultiScale(img, 1.3,4, cv2.cv.CV_HAAR_SCALE_IMAGE, (50,40))

	if len(rects) == 0:
		return [], img
	rects[:,2:] += rects[:, :2]
	return rects, img

def box(rects, img):
	for x1,y1,x2,y2 in rects:
		cv2.rectangle(img, (x1,y1),(x2,y2), (127,255,0),2)

cap = cv2.VideoCapture(0)
cap.set(3,400)
cap.set(4,300)

while(True):
	ret, img = cap.read()
	rects, img = detect(img)
	box(rects, img)
	cv2.imshow("frame", img)
	if(cv2.waitKey(1) & 0xFF == ord('q')):
		break
