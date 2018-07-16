import cv2 as cv
import os
from matplotlib.image import imsave

def image_selection():

    rootdir = os.getcwd()
    file_dir = os.path.join(rootdir, 'faces')
    savepath = os.path.join(rootdir, 'detected_faces')


    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    id = 100

    for subdir, dirs, files in os.walk(file_dir):

        for file in files:

            im = cv.imread(os.path.join(subdir, file))
            faces = face_cascade.detectMultiScale(im, 1.3, 5)

            for (x, y, w, h) in faces:

                crop_img = im[y:y + h, x:x + w]
                crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
                imsave(os.path.join(savepath, '{}.jpg'.format(id)), crop_img)
                id+=1

                # Plot the images with detected faces
                # cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv.imshow('img', im)
                # cv.waitKey(0)
                # cv.destroyAllWindows()


