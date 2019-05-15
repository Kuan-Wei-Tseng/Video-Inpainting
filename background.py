import numpy as np
import cv2

vid = cv2.VideoCapture("highway.mp4")

ret,oldframe = vid.read()
old_gray_frame = cv2.cvtColor(oldframe, cv2.COLOR_BGR2GRAY)
old_blur_frame = cv2.GaussianBlur(old_gray_frame,(5,5),0)
kernel = np.ones((15,15),np.uint8)

while True:
	ret,newframe = vid.read()
	new_gray_frame = cv2.cvtColor(newframe, cv2.COLOR_BGR2GRAY)
	new_blur_frame = cv2.GaussianBlur(new_gray_frame,(5,5),0)
	differ = cv2.absdiff(old_blur_frame,new_blur_frame)
	differ = cv2.morphologyEx(differ, cv2.MORPH_OPEN, kernel)
	differ = cv2.morphologyEx(differ, cv2.MORPH_CLOSE, kernel)
	# ret,differ = cv2.threshold(differ,25,255,cv2.THRESH_BINARY)
	ips, jps = np.where(differ > 8)

	for (ip,jp) in zip(ips,jps) :
		newframe[ip][jp][0] = 0
		newframe[ip][jp][1] = 255
		newframe[ip][jp][2] = 0

	cv2.imshow("differ",newframe)
	if cv2.waitKey(1) == 27:
		break
	oldframe = newframe
	old_gray_frame = new_gray_frame
	old_blur_frame = new_blur_frame

vid.release()
cv2.destroyAllWindows()






