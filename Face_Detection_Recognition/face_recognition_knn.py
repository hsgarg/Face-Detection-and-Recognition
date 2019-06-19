## RECOGNISE FACE USING KNN CLASSIFICATION ALGORITHM
# 1. load the training data (numpy arrays of all the persons)
		# x- values are stored in the numpy arrays
		# y-values needed to be assigned for each person
# 2. Read a video stream using opencv
# 3. Extract faces out of it, these faces will now be for testing purposes for which we want to predict the label
# 4. Use knn to find the prediction of face (int)
# 5. Map the predicted id to name of the user 
# 6. Display the predictions on the screen - bounding box and name

##### GOAL OF ALGORITHM : Given a new image, we want to see with whose face it resembles the most ########

##### FACE RECOGNITION #####

import cv2
import numpy as np 
import os 

############## KNN ALGORITHM ##################
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())

def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Get the vector and label
		ix = train[i, :-1]  ## We have used rest of the columns except last column for features
		iy = train[i, -1]   ## We have used last column for labels
		# Compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# Sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Retrieve only the labels
	labels = np.array(dk)[:, -1]
	
	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	# Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]

#################################################


## Initialize Camera
cap = cv2.VideoCapture(0)

## Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

skip = 0
dataset_path = './data/'

face_data = [] ## x values of data
labels = []   ## y values of data

class_id = 0 # Labels for the given file, 1st File that is loaded will have id=0, next will have id=1 and so on....
names = {}  #  Mapping between id and name


# Data Preparation
for fx in os.listdir(dataset_path):  ## iterating over directory
	if fx.endswith('.npy'):
		#Create a mapping btw class_id and name
		names[class_id] = fx[:-4] ##  taking all characters before .npy
		print("Loaded "+fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)   

		## Create Labels for the class
		## for each training point, in each file, we are omputing one label using this ->
		target = class_id*np.ones((data_item.shape[0],)) 
		class_id += 1
		labels.append(target)

## Concatenating all the items of the list into a single list
face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis=0).reshape((-1,1))

print(face_dataset.shape)
print(face_labels.shape)

## KNN Accepts accepts one training matrix in which we should have the x-data and y-data combined in a single matrix
trainset = np.concatenate((face_dataset,face_labels),axis=1)
print('Last Column for Labels and rest of the columns for Features : ')
print(trainset.shape)

############### TESTING ##################### 

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)
	if(len(faces)==0):
		continue

	for face in faces:
		x,y,w,h = face

		## Extracting Face Region of Interest
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		#Predicted Label (out)
		out = knn(trainset,face_section.flatten())

		## Display the name and rectangle around it on the Screen
		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(0,127,255),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,204),2)

	cv2.imshow("Faces",frame)

	key = cv2.waitKey(1) & 0xFF
	if key==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()




