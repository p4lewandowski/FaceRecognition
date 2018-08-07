import cv2 as cv
import os
from PyQt5.QtGui import QImage, QPixmap
import numpy as np

scale_factor = 1.15
min_neighbors = 3
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
cap.set(3, 640)  # WIDTH
cap.set(4, 480)  # HEIGHT


def image_selection():
    """Select images where faces are detected, crop and normalize them and save them to
    savepath directory."""

    rootdir = os.getcwd()
    file_dir = os.path.join(rootdir, '..', 'Data', 'raw_faces_data')
    savepath = os.path.join(file_dir, '..', 'detected_faces')

    id = 100

    for subdir, dirs, files in os.walk(file_dir):

        for file in files:

            im = cv.imread(os.path.join(subdir, file), 0)
            faces = face_cascade.detectMultiScale(im, scale_factor, min_neighbors)

            for (x, y, w, h) in faces:
                crop_img = im[y:y + h, x:x + w]
                crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
                crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)
                cv.imwrite(os.path.join(savepath, '{}_{}.pgm'.format(id, subdir.split('\\s')[1])),
                           crop_img)
                id += 1

                # Plot the images with detected faces
                # cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv.imshow('img', im)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

def detect_face(filepath):
    im = cv.imread(filepath, 0)
    faces = face_cascade.detectMultiScale(im, scale_factor, min_neighbors)
    for (x, y, w, h) in faces:
        cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow('img', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

def image_cropping(im = None, cnn=False, filepath = None, save=False):
    """Read in the image either from file or from image frame from a camera.
    Either save it to file or just return the image."""
    if filepath:
        im = cv.imread(filepath, 0)

    else:
        faces = face_cascade.detectMultiScale(im, scale_factor, min_neighbors)
        if np.any(faces):
            # Take the biggest found face
            (x, y, w, h) = faces[0]
            for i, coord in enumerate(faces):
                if coord[2] > w:
                    (x, y, w, h) = coord

            # If cnn is used we require bigger image, not only face
            if cnn:
                bias = 15
                crop_img = im[y-bias:y + h + bias, x-bias:x + w + bias]
                crop_img = cv.resize(crop_img, dsize=(180, 180), interpolation=cv.INTER_CUBIC)
                crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)
                crop_img = crop_img / 255.

            else:
                crop_img = im[y:y + h, x:x + w]
                crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
                crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)

        if len(faces) == 0:
            return False


    if save:
        if filepath:
            path = ''.join(filepath.split('/')[:-1])
            filename = filepath.split('/')[-1].split('.')[0]
            cv.imwrite(os.path.join(path, '{}_newface.pgm'.format(filename)), crop_img)
        else:
            cv.imwrite('newface.pgm', crop_img)

    return crop_img


def face_recording(gui=False, im_count=20, idle=True, cnn=False):
    """Take images from camera and return array with all the cropped images collected where face was detected.
        gui - if displaying in gui enabled
        im_count - how many images to take
        idle - if enabled - give some 'free' frames before image processing takes place"""

    face_data = []
    # Enable some idle time (number of frames without computations)
    if idle:
        idle = 20

    while (len(face_data) < im_count):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        # If there are any faces at all
        for (x, y, w, h) in faces:
            # Display rectangle where face is detected
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            if not idle:
                im = image_cropping(im=gray, save=False, cnn=cnn)
                face_data.append(im)

        if idle:
            idle -= 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if gui:
            RGB_image = np.array(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            image = QImage(
                RGB_image,
                RGB_image.shape[1],
                RGB_image.shape[0],
                RGB_image.shape[1] * 3,
                QImage.Format_RGB888
            )
            gui.AddPersonLabel.setPixmap(QPixmap.fromImage(image))
        else:
            cv.imshow('frame', frame)

    cv.destroyAllWindows()

    if gui:
        gui.AddPersonLabel.clear()

    return face_data

def take_image(gui=False, idle=True, cnn=False):
    """The closest, biggest face will be considered"""
    if idle:
        idle=20

    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if idle:
            idle -= 1

        if gui:
            RGB_image = np.array(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
            image_gui = QImage(
                RGB_image,
                RGB_image.shape[1],
                RGB_image.shape[0],
                RGB_image.shape[1] * 3,
                QImage.Format_RGB888
            )
            gui.IdentifySearchLabel.setPixmap(QPixmap.fromImage(image_gui))
        else:
            cv.imshow('frame', frame)

        if not idle:
            image = image_cropping(im=gray)
            if image is not False:
                cv.destroyAllWindows()
                break

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    if gui:
        gui.IdentifySearchLabel.clear()

    return image



if __name__ == "__main__":
    face_recording(cnn=True)
    take_image()




