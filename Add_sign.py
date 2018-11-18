# organize imports
import cv2
import imutils
import numpy as np
from sklearn.metrics import pairwise
import pickle, os, sqlite3, random

image_x, image_y = 50, 50

def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def init_create_folder_database():
	# create the folder and database if not exist
	if not os.path.exists("gestures"):
		os.mkdir("gestures")
	if not os.path.exists("gesture_db.db"):
		conn = sqlite3.connect("gesture_db.db")
		create_table_cmd = "CREATE TABLE gesture ( g_id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT UNIQUE, g_name TEXT NOT NULL )"
		conn.execute(create_table_cmd)
		conn.commit()

def create_folder(folder_name):
	if not os.path.exists(folder_name):
		os.mkdir(folder_name)

def store_in_db(g_id, g_name):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "INSERT INTO gesture (g_id, g_name) VALUES (%s, \'%s\')" % (g_id, g_name)
	try:
		conn.execute(cmd)
	except sqlite3.IntegrityError:
		choice = input("g_id already exists. Want to change the record? (y/n): ")
		if choice.lower() == 'y':
			cmd = "UPDATE gesture SET g_name = \'%s\' WHERE g_id = %s" % (g_name, g_id)
			conn.execute(cmd)
		else:
			print("Doing nothing...")
			return
	conn.commit()
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
	


#-------------------------------------------------------------------------------
# Main function
#-------------------------------------------------------------------------------
def store_images(g_id):
	# initialize accumulated weight
	accumWeight = 0.5
	total_pics = 1200

	# get the reference to the webcam
	camera = cv2.VideoCapture(0)
	cap = cv2.VideoCapture('testvideo.mp4')
	flag=0
	paused=0
	cntframe=0
	# region of interest (ROI) coordinates
	top, right, bottom, left = 10, 350, 225, 590
	create_folder("gestures/"+str(g_id))
	pic_no = -50
	flag_start_capturing = False
	frames = 0

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
		#input('Press Enter to continue')
		if num_frames < 30:
			run_avg(gray, accumWeight)
			if num_frames == 1:
				print("[STATUS] please wait! calibrating...")
			elif num_frames == 29:
				print("[STATUS] Successfull...")
			#input('Press Enter to continue')
		else:
			# segment the hand region
			hand = segment(gray)
			if hand is None:
				if flag==1 and paused==0:
					ret, framevid = cap.read()
					dst = cv2.resize(framevid, None, fx = 0.5, fy=0.5)
					cv2.imshow("Display frame", dst);
				
				if flag==1 and paused==1:
					cv2.imshow("Display frame", dst)

			# check whether hand region is segmented
			if hand is not None:
				# if yes, unpack the thresholded image and
				# segmented region
				(thresholded, segmented) = hand

				# draw the segmented region and display the frame
				cv2.drawContours(clone, [segmented + (right, top)], -1, (0, 0, 255))

				# count the number of fingers
				
				# show the thresholded image
				cv2.imshow("Thesholded", thresholded)
				# display the frame with segmented hand
				cv2.rectangle(clone, (left, top), (right, bottom), (0,255,0), 2)
				cv2.putText(clone, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (200, 127, 255))
				cv2.imshow("Video Feed", clone)
				save_img = thresholded
				cv2.putText(clone, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
				cv2.imwrite("gestures/"+str(g_id)+"/"+str(pic_no)+".jpg", save_img)
				keypress = cv2.waitKey(1)
				if keypress == ord('c'):
					if flag_start_capturing == False:
						flag_start_capturing = True
					else:
						flag_start_capturing = False
						frames = 0
				if pic_no == 0:
					
					input('Press Enter to start')
				pic_no += 1
				if pic_no == total_pics:
					break
		# draw the segmented hand
		

		# increment the number of frames
		num_frames += 1

		

			
init_create_folder_database()
g_id = input("Enter gesture no.: ")
g_name = input("Enter gesture name/text: ")
store_in_db(g_id, g_name)
store_images(g_id)
