import numpy as np
import cv2 as cv
import os


im = cv.imread(r'BioID_0001.pgm')

fname='pobrane.jpg'
fname="p.jpg"
img = cv.imread(fname)
# im = img
# im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)


face_cascade = cv.CascadeClassifier(r'C:\Users\Aniesia\PycharmProjects\FaceRecognition\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')

faces = face_cascade.detectMultiScale(im, 1.3, 5)
for (x,y,w,h) in faces:
    cv.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)

cv.imshow('img',im)
cv.waitKey(0)
cv.destroyAllWindows()