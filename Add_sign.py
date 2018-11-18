import cv2
import numpy as np
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
	
def store_images(g_id):
	total_pics = 1200
	camera = cv2.VideoCapture(0)
	cap = cv2.VideoCapture('testvideo.mp4')
	flag=0
	paused=0
	cntframe=0
	# region of interest (ROI) coordinates
	top, right, bottom, left = 10, 350, 225, 590

	create_folder("gestures/"+str(g_id))
	pic_no = 0
	flag_start_capturing = False
	frames = 0
	
	while True:
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
        pic_no += 1
        save_img = thresh[y1:y1+h1, x1:x1+w1]
				if w1 > h1:
					save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
				elif h1 > w1:
					save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
				save_img = cv2.resize(save_img, (image_x, image_y))
				rand = random.randint(0, 10)
				if rand % 2 == 0:
					save_img = cv2.flip(save_img, 1)
				cv2.putText(img, "Capturing...", (30, 60), cv2.FONT_HERSHEY_TRIPLEX, 2, (127, 255, 255))
				cv2.imwrite("gestures/"+str(g_id)+"/"+str(pic_no)+".jpg", save_img)

		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		cv2.putText(img, str(pic_no), (30, 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (127, 127, 255))
		cv2.imshow("Capturing gesture", img)
		cv2.imshow("thresh", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('c'):
			if flag_start_capturing == False:
				flag_start_capturing = True
			else:
				flag_start_capturing = False
				frames = 0
		if flag_start_capturing == True:
			frames += 1
		if pic_no == total_pics:
			break

init_create_folder_database()
g_id = input("Enter gesture no.: ")
g_name = input("Enter gesture name/text: ")
store_in_db(g_id, g_name)
store_images(g_id)
