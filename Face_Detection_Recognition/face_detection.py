## Face Detection using OpenCV and HaarCascades
## Haarcascade Classifier is already trained on a lot of facial data

############# Once done detecting -> PRESS 'q' to BREAK THE LOOP #################

import cv2


##1 Capturing the device from which we want to read our video stream, 0 is for default webcam
cap = cv2.VideoCapture(0)

##2 Creating a clssifier object
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")


while True:
	## cap.read() is a method which returns 2 a boolean value and the frame that is captured
	## if the boolean value is false, it means the frame is not captured properly

	ret,frame = cap.read()
	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	if ret == False:
		continue

	##3 detectMultiScale : this method will return starting coordinates(x,y), width and height
	## If there are multiple faces present, it will return a list of tuples like [(x,y,w,h),(x1,y1,w1,h1),(),....]
	## Here 1.3 is the 	Scale Factor and 5 is the number of Neighbours
	## Reason for Scaling : The Kernals should operate on similar size images on whih they are trained on
	## 3~6 is a good value for Number of Neighbours, hence we have given 5
	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)


	##4 Iterate over the list and we are going to draw a bounding box around each face
	for (x,y,w,h) in faces:

		## 2 end pts of the rectangle : (x,y) and (x+w,y+h)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,250,0),2)
		cv2.rectangle(gray_frame,(x,y),(x+w,y+h),(255,0,0),2)

	## Displaying Color Frame and Gray Frame
	cv2.imshow("BGR Frame",frame)
	cv2.imshow("Gray Frame", gray_frame)


	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()


### scaleFactor – It Specifies how much the image size is reduced at each image scale.

'''
minNeighbors – It specifies how many neighbors each candidate rectangle should have to retain it.
This parameter will affect the quality of the detected faces. 
'''
