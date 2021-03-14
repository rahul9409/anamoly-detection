#!/usr/bin/env python
from imutils import face_utils
import dlib
import cv2


import numpy as np
def curve(points):
	x = points[:,0]
	y = points[:,1]

	z = np.polyfit(x, y, 2)
	f = np.poly1d(z)

	x_n = np.linspace(x[0], x[-1], 100)
	y_n = f(x_n)
	return list(zip(x_n, y_n))

#Loading the model for detection

#from curve_fitting import curve
from imutils import face_utils
import dlib
import cv2

img_path='1.png'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#convert to grayscale
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = detector(gray)

#Printing location of all the facial features present
for (j,face) in enumerate(faces):
        points = predictor(gray, face)
        print('\nface #',j+1)
        l=[]
        print('\nface boundary coordinate\n')
        for i in range(0, 27):  # loop for displaying face boundary coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)
        print('\nnose coordinate\n')
        for i in range(27, 36):  # loop for displaying nose  coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)
        print('\nleft eye coordinate\n')
        for i in range(36, 42):  # loop for displaying left eye coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)
        print('\nright eye coordinate\n')
        for i in range(42, 48):  # loop for displaying right eye coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)
        print('\nlips coordiante\n')
        for i in range(48, 68):  # loop for displaying lips coordinate
            curr_c = (points.part(i).x, points.part(i).y)
            print(curr_c)

        for i in range(5, 12):                          #loop for storing jaw coordinates
            curr_c=(points.part(i).x, points.part(i).y)
            l.append(curr_c)

        cur=np.array(curve(np.array(l)), np.int32)      # calling function to trace proper fitting curve


        for i in range(len(cur)-1):                          #loop for tracing jaw line
            curr_c=(cur[i][0], cur[i][1])
            next_cordi=(cur[i+1][0], cur[i+1][1])
            cv2.line(img, curr_c, next_cordi, (0, 0, 0), 3)
        for n in range(0, 68):                          #loop for detecting feature points on face
        	x = points.part(n).x
        	y = points.part(n).y
        	cv2.circle(img, (x, y), 3, (0, 0, 255), 2)

        #points = face_utils.shape_to_np(points)

        # to  convert dlib's rectangle to a OpenCV-style bounding box
        # [i.e., (x, y, w, h)], then draw the face bounding box
        (x, y, w, h) = face_utils.rect_to_bb(face)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # to display the face number
        cv2.putText(img, "Face #{}".format(j + 1), (x - 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#cv2.imwrite(path, img)
cv2.imwrite("faces_featureimg.jpg", img) #writes image with landmark points
