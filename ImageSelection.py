import cv2 as cv
import os
from matplotlib.image import imsave

rootdir = 'C:\\Users\\Aniesia\\PycharmProjects\\FaceRecognition'
face_cascade = cv.CascadeClassifier(r'C:\Users\Aniesia\PycharmProjects\FaceRecognition\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
id = 0
savepath = 'faces\\detected_faces'

for subdir, dirs, files in os.walk(rootdir):

    for file in files:

        im = cv.imread(os.path.join(subdir, file))
        #print(os.path.join(subdir, file))
        faces = face_cascade.detectMultiScale(im, 1.3, 5)

        for (x, y, w, h) in faces:

            crop_img = im[y:y + h, x:x + w]
            crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
            imsave(os.path.join(savepath, '{}.jpg'.format(id)), crop_img)
            id+=1

