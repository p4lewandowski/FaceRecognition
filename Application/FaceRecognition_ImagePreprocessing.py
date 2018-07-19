import cv2 as cv
import os
from matplotlib.image import imsave
import matplotlib.pyplot as plt

def image_selection():
    """Select images where faces are detected, crop and normalize them and save them to
    savepath directory."""

    rootdir = os.getcwd()
    file_dir = os.path.join(rootdir, '..', 'Data', 'raw_faces_data')
    savepath = os.path.join(file_dir, '..', 'detected_faces')

    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

    id = 100

    for subdir, dirs, files in os.walk(file_dir):

        for file in files:

            im = cv.imread(os.path.join(subdir, file), 0)
            faces = face_cascade.detectMultiScale(im, 1.3, 7)

            for (x, y, w, h) in faces:
                crop_img = im[y:y + h, x:x + w]
                crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
                crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)
                #imsave(os.path.join(savepath, '{}.jpg'.format(id)), crop_img, cmap=plt.cm.bone)
                cv.imwrite(os.path.join(savepath, '{}_{}.pgm'.format(id, subdir.split('\\s')[1])),
                           crop_img)
                id += 1

                # Plot the images with detected faces
                # cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv.imshow('img', im)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

def image_cropping(filepath, findface = False):
    rootdir = os.getcwd()
    newdir = os.path.join(rootdir, '..', 'Data', 'new_faces_notindetected')
    im = cv.imread(os.path.join(newdir, filepath), 0)
    print(os.path.join(newdir, filepath))

    if findface:
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(im, 1.3, 7)
        for (x, y, w, h) in faces:
            crop_img = im[y:y + h, x:x + w]

    else:
        crop_img = im[10:100, 2:90]

    crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
    crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)
    cv.imwrite(os.path.join(newdir, '{}_newface.pgm'.format(
        filepath.split('/')[-1].split('.')[0])), crop_img)

    return crop_img

def detect_face(filepath):
    im = cv.imread(filepath, 0)
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(im, 1.05, 3)
    for (x, y, w, h) in faces:
        cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow('img', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    # image_selection()
    # image_cropping('s5', '1.pgm')
    detect_face(r'C:\Users\Aniesia\PycharmProjects\FaceRecognition\Data\\new_faces_notindetected\ja1.jpg')





