# -*- coding: utf-8 -*-
import face_recognition
import numpy
import cv2
from datetime import datetime
image = cv2.imread("faces2.jpg")


# Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
frame = image[:, :, ::-1]

face_locations = face_recognition.face_locations(frame)
face_encodings = face_recognition.face_encodings(frame, face_locations)

"""When faces doesn't repeat"""
print(str(len(face_encodings))+" analisadas")
i = 0
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    print("Salvando face "+str(i + 1))
    image_name = "saved_ims/face "+str(i + 1)+".jpg"
    cv2.imwrite(image_name,image[top:bottom,left:right])
    i = i + 1
    
"""When faces repeats but save just one"""

j = 0
faces_analyzed = []
matches = [False]

for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    if(j != 0):
        for face_encoded in faces_analyzed:
            matches = face_recognition.compare_faces(face_encoding,[face_encoded])
            if(matches[0]):
                break
    if(matches[0] != True):
        print("Salvando face "+str(j + 1))
        image_name = "saved_ims/face "+str(j + 1)+".jpg"
        cv2.imwrite(image_name,image[top:bottom,left:right])
        faces_analyzed.append(face_encoding)
        j = j + 1
print("Total de "+str(j+1))