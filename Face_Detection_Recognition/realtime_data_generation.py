# This is a Python Script that 
# 	1. Captures images from the webcam video stream
# 	2. Extracts all Faces from the image frame (using haarcascades)
# 	3. Stores the Face information into numpy arrays
#		-> Take the LARGEST Face (in case of multiple faces)
#		-> Crop this face and create a 2-D matrix(this is the image itself)
#		-> Flatten it in the form of a Linear Array and save it as a .np file

################## STEPS FOLLOWED ###################

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

######### WE are generating Real Time Training Data by Taking Selfies from Webcam #########

######## Evertime this script is run, One File will be created in the Data Folder #########


import cv2
import numpy as np

## Initialize the camera
cap = cv2.VideoCapture(0)

## Detcting Face
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
##Storing the data. hence creating an array 
face_data = []
dataset_path = './data/'  ##data is a folder inside the project folder

## Taking File Name as Input From the User
file_name = input("Enter the name of the person : ")

while True:
	ret,frame = cap.read()

	if ret==False:
		continue

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	
	## faces will be a list of Tuples [(x,y,widthh,height),(x1,y1,w1,h1),....]
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if len(faces)==0:
		continue

	## Sorting on the basis of area, f[2] = w and f[3] = h hence area = w*h = f[2]*f[3]
	## If we do reverse = True, largest will come to start
	faces = sorted(faces,key=lambda f:f[2]*f[3])

	##Draw bounding box
	##Pick the last face, because its the largest in the area
	for face in faces[-1:]:
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		##Crop out required part : Region of interest
		## Add offset of 10px on each side of face
		offset = 10
		## SLICING
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		## Resizing the face_section array into 100x100
		face_section = cv2.resize(face_section,(100,100))

		skip += 1

		## Store every 10th frame
		if skip%10==0:
			face_data.append(face_section)
			## how many faces we have captured so far
			print(len(face_data))

	## Show Frame
	cv2.imshow("Frame",frame)

	##Show the face
	cv2.imshow("Face Section",face_section)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert the face list array into a numpy array
face_data = np.asarray(face_data)
## Number of rows should be same as number of faces ace_data.shape[0]
## Number of cols should be decided automatically hence we have given -1

face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)
 
## SAVE THIS DATA INTO FILE SYSTEM
## file_name is the user input
## face_data is the numpy array that is to be saved
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully save at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()

### You will see 2 windows, one is the frame and one is only the face
