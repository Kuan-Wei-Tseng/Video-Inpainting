import numpy as np
import cv2
from PIL import Image

vid = cv2.VideoCapture("sbadminton.mp4")

ret,old_frame = vid.read()
old_gray_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
old_blur_frame = cv2.GaussianBlur(old_gray_frame,(5,5),0)
kernel1 = np.ones((5,5),np.uint8)
kernel2 = np.ones((15,15),np.uint8)

M,N = old_gray_frame.shape
counter = 0
color = 'badminton/frames'
mask = 'badminton/masks'

while True:
	# frame counter:
	counter = counter + 1

	ret,new_frame = vid.read()
	ret,new_frame = vid.read()

	if not ret:
		break
	if counter < 2 or counter > 200:
		continue

	new_gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
	new_blur_frame = cv2.GaussianBlur(new_gray_frame,(5,5),0)
	differ = cv2.absdiff(old_blur_frame,new_blur_frame)

	# Binarize the Difference Matrix and process it with Morphological techniques.
	ret,fildiff = cv2.threshold(differ,3,255,cv2.THRESH_BINARY)
	fildiff = cv2.morphologyEx(fildiff, cv2.MORPH_OPEN, kernel1)
	fildiff = cv2.morphologyEx(fildiff, cv2.MORPH_CLOSE, kernel2)

	# Connected Component Labeling
	ret, new_label, new_stats, _= cv2.connectedComponentsWithStats(fildiff,8,cv2.CV_32S)

	# Visualization:
	viewer = np.copy(new_frame)
	new_marker = np.zeros((M,N)).astype(np.uint8)

	obj_counter = 0;
	motion = 0

	for obj in new_stats:
		# Skip the background label:
		obj_counter = obj_counter + 1
		if obj_counter == 1:
			continue
		# Skip the small moving block:
		if obj[4] < 200:
			continue

		motion = 1

		# Mark each objects on the image (based on motion)
		cv2.rectangle(new_marker,(obj[0],obj[1]),(obj[0]+obj[2],obj[1]+obj[3]),128, -1)
		cv2.rectangle(viewer,(obj[0],obj[1]),(obj[0]+obj[2],obj[1]+obj[3]),(0, 255, 0), 2)
		cv2.rectangle(new_label,(obj[0],obj[1]),(obj[0]+obj[2],obj[1]+obj[3]),obj_counter,-1)

		# Target frames for motion + search:
		if counter >= 3 and counter <= 700 and counter % 15 != 0:

			# Crop the patch image:
			patch = new_frame[obj[1]:obj[1]+obj[3],obj[0]:obj[0]+obj[2]]
			w,h,_ = patch.shape

			# Perform Template Matching:
			res = cv2.matchTemplate(old_frame,patch,cv2.TM_CCORR_NORMED)
			min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

			# Coordinate of matched blocks on old image frame
			top_left = max_loc  # topleft = (h,v)
			bottom_right = (top_left[0] + h, top_left[1] + w)

			# Draw the matched patch on a blank marker
			patch_marker = np.zeros((M,N)).astype(np.uint8)
			cv2.rectangle(patch_marker,top_left, bottom_right, 128, -1)

			# Bitwise "and" to find the intersection of old_marker and patch_marker
			result = patch_marker & old_marker
			rx,ry = np.where(result == 128)
			'''
			if counter == 510:
				print(counter)
				cv2.imshow("patch",patch)
				cv2.waitKey(0)
			else:
				print(counter)
			'''
			# s = np.copy(old_marker)
			
			if len(rx) > 0.5 * w * h:
				# s = np.copy(old_marker)
				# cv2.rectangle(s,top_left, bottom_right, 255, 2)
				BL = old_label[rx[0]][ry[0]]

				'''
				if counter == 510:
					print(counter)
					print("Old label",BL)
					print("Old Stats",old_stats)
					cv2.imshow("result",result)
					cv2.waitKey(0)
				'''
				try:
					if BL != 0:
						cv2.rectangle(new_marker,(old_stats[BL][0],old_stats[BL][1]),(old_stats[BL][0]+old_stats[BL][2],old_stats[BL][1]+old_stats[BL][3]),128, -1)
						cv2.rectangle(viewer,(old_stats[BL][0],old_stats[BL][1]),(old_stats[BL][0]+old_stats[BL][2],old_stats[BL][1]+old_stats[BL][3]),(255, 0, 0), 2)
				except:
					continue


			#else:
				# cv2.rectangle(s,top_left, bottom_right, 50, 2)
			
			#cv2.imshow("old_marker",s)
			#cv2.waitKey(0)
			#cv2.imshow("result",result)
			#if cv2.waitKey(0) == 27:
			#	quit()

	if motion == 0 and counter >=3:
		new_marker = np.copy(old_marker)
		viewer = np.copy(viewer)

	print("frame number",counter)

	cv2.imshow('viewer',viewer)
	if cv2.waitKey(500) == 27:
		break

	'''
	if counter == 508 or counter == 509:
		cv2.imwrite('../timg'+str(counter)+'.bmp',new_frame)
		cv2.imshow('marker',new_marker)
		cv2.waitKey(0)
	'''
	old_frame = np.copy(new_frame)
	old_marker = np.copy(new_marker)
	old_gray_frame = np.copy(new_gray_frame)
	old_blur_frame = np.copy(new_blur_frame)

	# Re-connected component labeling:
	ret, old_label, old_stats, _= cv2.connectedComponentsWithStats(old_marker,8,cv2.CV_32S)

	mfname = mask + str(counter) + '.bmp'
	cfname = color + str(counter) + '.bmp'
	
	cv2.imwrite(cfname,new_frame)
	cv2.imwrite(mfname,new_marker)

vid.release()
cv2.destroyAllWindows()












