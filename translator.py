

import cv2, pickle
import numpy as np
import tensorflow as tf
from cnn_tf import cnn_model_fn
import os
import sqlite3
from keras.models import load_model
import imutils
#import numpy as np
from sklearn.metrics import pairwise


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
classifier = tf.estimator.Estimator(model_dir="tmp/cnn_model2", model_fn=cnn_model_fn)
prediction = None
model = load_model('cnn_model_keras2.h5')


# global variables
bg = None

#-------------------------------------------------------------------------------
# Function - To find the running average over the background
#-------------------------------------------------------------------------------
def run_avg(image, accumWeight):
	global bg
	# initialize the background
	if bg is None:
		bg = image.copy().astype("float")
		return

	# compute weighted average, accumulate it and update the background
	cv2.accumulateWeighted(image, bg, accumWeight)

#-------------------------------------------------------------------------------
# Function - To segment the region of hand in the image
#-------------------------------------------------------------------------------
def segment(image, threshold=25):
	global bg
	# find the absolute difference between background and current frame
	diff = cv2.absdiff(bg.astype("uint8"), image)

	# threshold the diff image so that we get the foreground
	thresholded = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)[1]

	# get the contours in the thresholded image
	(_, cnts, _) = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# return None, if no contours detected
	if len(cnts) == 0:
		return
	else:
		# based on contour area, get the maximum contour which is the hand
		segmented = max(cnts, key=cv2.contourArea)
		return (thresholded, segmented)

#-------------------------------------------------------------------------------
# Function - To count the number of fingers in the segmented hand region
#-------------------------------------------------------------------------------
def count(thresholded, segmented):
	# find the convex hull of the segmented hand region
	chull = cv2.convexHull(segmented)

	# find the most extreme points in the convex hull
	extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
	extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
	extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
	extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

	# find the center of the palm
	cX = (extreme_left[0] + extreme_right[0]) / 2
	cY = (extreme_top[1] + extreme_bottom[1]) / 2

	# find the maximum euclidean distance between the center of the palm
	# and the most extreme points of the convex hull
	distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
	maximum_distance = distance[distance.argmax()]
	
	# calculate the radius of the circle with 80% of the max euclidean distance obtained
	radius = int(0.8 * maximum_distance)
	
	# find the circumference of the circle
	circumference = (2 * np.pi * radius)

	# take out the circular region of interest which has 
	# the palm and the fingers
	circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
	# draw the circular ROI
	cv2.circle(circular_roi, (int(cX),int(cY)), radius, 255, 1)
	
	# take bit-wise AND between thresholded hand using the circular ROI as the mask
	# which gives the cuts obtained using mask on the thresholded hand image
	circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

	# compute the contours in the circular ROI
	(_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# initalize the finger count
	count = 0

	# loop through the contours found
	for c in cnts:
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)

		# increment the count of fingers only if -
		# 1. The contour region is not the wrist (bottom area)
		# 2. The number of points along the contour does not exceed
		#     25% of the circumference of the circular ROI
		if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
			count += 1

	return count

	
def saveimage(thresholded, segmented):
	# find the convex hull of the segmented hand region
	chull = cv2.convexHull(segmented)

	# find the most extreme points in the convex hull
	extreme_top    = tuple(chull[chull[:, :, 1].argmin()][0])
	extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
	extreme_left   = tuple(chull[chull[:, :, 0].argmin()][0])
	extreme_right  = tuple(chull[chull[:, :, 0].argmax()][0])

	# find the center of the palm
	cX = (extreme_left[0] + extreme_right[0]) / 2
	cY = (extreme_top[1] + extreme_bottom[1]) / 2

	# find the maximum euclidean distance between the center of the palm
	# and the most extreme points of the convex hull
	distance = pairwise.euclidean_distances([(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom])[0]
	maximum_distance = distance[distance.argmax()]
	
	# calculate the radius of the circle with 80% of the max euclidean distance obtained
	radius = int(0.8 * maximum_distance)
	
	# find the circumference of the circle
	circumference = (2 * np.pi * radius)

	# take out the circular region of interest which has 
	# the palm and the fingers
	circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
	
	# draw the circular ROI
	cv2.circle(circular_roi, (int(cX),int(cY)), radius, 255, 1)
	
	# take bit-wise AND between thresholded hand using the circular ROI as the mask
	# which gives the cuts obtained using mask on the thresholded hand image
	circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)

	# compute the contours in the circular ROI
	(_, cnts, _) = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

	# initalize the finger count
	count = 0

	# loop through the contours found
	for c in cnts:
		# compute the bounding box of the contour
		(x, y, w, h) = cv2.boundingRect(c)

		# increment the count of fingers only if -
		# 1. The contour region is not the wrist (bottom area)
		# 2. The number of points along the contour does not exceed
		#     25% of the circumference of the circular ROI
		if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
			count += 1

	return count
	


def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def tf_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	np_array = np.array(img)
	return np_array

def tf_predict(classifier, image):
	'''
	need help with prediction using tensorflow
	'''
	global prediction
	processed_array = tf_process_image(image)
	pred_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x":processed_array}, shuffle=False)
	pred = classifier.predict(input_fn=pred_input_fn)
	prediction = next(pred)
	print(prediction)

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def split_sentence(text, num_of_words):
	'''
	Splits a text into group of num_of_words
	'''
	list_words = text.split(" ")
	length = len(list_words)
	splitted_sentence = []
	b_index = 0
	e_index = num_of_words
	while length > 0:
		part = ""
		for word in list_words[b_index:e_index]:
			part = part + " " + word
		splitted_sentence.append(part)
		b_index += num_of_words
		e_index += num_of_words
		length -= num_of_words
	return splitted_sentence

def put_splitted_text_in_blackboard(blackboard, splitted_text):
	y = 200
	for text in splitted_text:
		cv2.putText(blackboard, text, (4, y), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		y += 50

def get_hand_hist():
	with open("hist", "rb") as f:
		#hist = pickle.load(f)
		u = pickle._Unpickler(f)
		u.encoding = 'latin1'
		hist = u.load()
	return hist


def recognize():
# initialize accumulated weight
	accumWeight = 0.5

	# get the reference to the webcam
	camera = cv2.VideoCapture(0)
	cap = cv2.VideoCapture('testvideo.mp4')
	flag=0
	paused=0
	cntframe=0
	# region of interest (ROI) coordinates
	top, right, bottom, left = 10, 350, 225, 590

	# initialize num of frames
	num_frames = 0

	# calibration indicator
	calibrated = False

	# keep looping, until interrupted
	while(True):
		# get the current frame
		(grabbed, frame) = camera.read()

		# resize the frame
		frame = imutils.resize(frame, width=700)

		# flip the frame so that it is not the mirror view
		frame = cv2.flip(frame, 1)

		# clone the frame
		clone = frame.copy()

		# get the height and width of the frame
		(height, width) = frame.shape[:2]

		# get the ROI
		roi = frame[top:bottom, right:left]

		# convert the roi to grayscale and blur it
		gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
		gray = cv2.GaussianBlur(gray, (7, 7), 0)

		# to get the background, keep looking till a threshold is reached
		# so that our weighted average model gets calibrated
		if num_frames < 30:
			run_avg(gray, accumWeight)
			if num_frames == 1:
				print("[STATUS] please wait! calibrating...")
			elif num_frames == 29:
				print("[STATUS] Successfull...")
		else:
			# segment the hand region
			hand = segment(gray)

			# check whether hand region is segmented
			if hand is not None:
				# if yes, unpack the thresholded image and
				# segmented region
				(thresholded, segmented) = hand

				# draw the segmented region and display the frame
				cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

				# count the number of fingers
				fingers = count(thresholded, segmented)
				cv2.imshow('image',thresholded)
				cntframe=cntframe+1
				
				if cntframe==40:
					if fingers==5:
						flag=1
					
					if flag==1 and fingers==2:
						paused=1
					
					if flag==1 and fingers==4:
						paused=0
				
					cntframe=0
					cv2.putText(clone, "Now", (70, 145), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
				

				pred_probab, pred_class = keras_predict(model, thresholded)
				print(pred_class, pred_probab)
				text = ""
				if pred_probab*100 > 80:
					text = get_pred_text_from_db(pred_class)
					print(text)
				if flag==1 and paused==0:
					ret, framevid = cap.read()
					dst = cv2.resize(framevid, None, fx = 0.5, fy=0.5)
					cv2.imshow("Display frame", dst);
				
				if flag==1 and paused==1:
					cv2.imshow("Display frame", dst)
					
				cv2.putText(clone, text, (70, 45), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
				
				# show the thresholded image
				cv2.imshow("Thresholded", thresholded)
				
			if hand is None:
				if flag==1 and paused==0:
					ret, framevid = cap.read()
					dst = cv2.resize(framevid, None, fx = 0.5, fy=0.5)
					cv2.imshow("Display frame", dst);
				
				if flag==1 and paused==1:
					cv2.imshow("Display frame", dst)
				
		# draw the segmented hand
		cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)

		# increment the number of frames
		num_frames += 1

		# display the frame with segmented hand
		cv2.imshow("Video Feed", clone)

		# observe the keypress by the user
		keypress = cv2.waitKey(1) & 0xFF

		# if the user pressed "q", then stop looping
		if keypress == ord("q"):
			break

	# free up memory
	camera.release()
	cv2.destroyAllWindows()
	

'''
	global prediction
	cam = cv2.VideoCapture(1)
	if cam.read()[0] == False:
		cam = cv2.VideoCapture(0)
	hist = get_hand_hist()
	x, y, w, h = 300, 100, 300, 300
	while True:
		text = ""
		img = cam.read()[1]
		img = cv2.flip(img, 1)
		imgCrop = img[y:y+h, x:x+w]
		imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
		disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
		cv2.filter2D(dst,-1,disc,dst)
		blur = cv2.GaussianBlur(dst, (11,11), 0)
		blur = cv2.medianBlur(blur, 15)
		thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
		thresh = cv2.merge((thresh,thresh,thresh))
		thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
		thresh = thresh[y:y+h, x:x+w]
		contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[1]
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			#print(cv2.contourArea(contour))
			if cv2.contourArea(contour) > 10000:
				x1, y1, w1, h1 = cv2.boundingRect(contour)
				save_img = thresh[y1:y1+h1, x1:x1+w1]
				
				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				
				pred_probab, pred_class = keras_predict(model, save_img)
				print(pred_class, pred_probab)
				
				if pred_probab*100 > 80:
					text = get_pred_text_from_db(pred_class)
					print(text)
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		splitted_text = split_sentence(text, 2)
		put_splitted_text_in_blackboard(blackboard, splitted_text)
		#cv2.putText(blackboard, text, (30, 200), cv2.FONT_HERSHEY_TRIPLEX, 1.3, (255, 255, 255))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, blackboard))
		cv2.imshow("Recognizing gesture", res)
		cv2.imshow("thresh", thresh)
		if cv2.waitKey(1) == ord('q'):
			break
	'''

keras_predict(model, np.zeros((50, 50), dtype=np.uint8))		
recognize()
