import cv2 as cv
import os
from PyQt5.QtGui import QImage, QPixmap
import matplotlib.pyplot as plt

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

def image_cropping(im = None, filepath = None, findface = False, save=False):
    """Read in the image either from file or from image frame from a camera.
    Either save it to file or just return the image."""
    if filepath:
        im = cv.imread(filepath, 0)

    if findface:
        face = face_cascade.detectMultiScale(im, scale_factor, min_neighbors)
        for (x, y, w, h) in face:
            crop_img = im[y:y + h, x:x + w]
        if len(face) == 0:
            return False

    else:
        crop_img = im[10:100, 2:90]


    crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
    crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)

    if save:
        if filepath:
            path = ''.join(filepath.split('/')[:-1])
            filename = filepath.split('/')[-1].split('.')[0]
            cv.imwrite(os.path.join(path, '{}_newface.pgm'.format(filename)), crop_img)
        else:
            cv.imwrite('newface.pgm', crop_img)

    return crop_img

def detect_face(filepath):
    im = cv.imread(filepath, 0)
    faces = face_cascade.detectMultiScale(im, scale_factor, min_neighbors)
    for (x, y, w, h) in faces:
        cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow('img', im)
    cv.waitKey(0)
    cv.destroyAllWindows()


def face_recording(gui=False):

    face_data = []

    while (len(face_data) < 20):
        # Capture frame-by-frame
        ret, frame = cap.read()
        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        # Display the resulting frame
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            im = image_cropping(im=gray, findface=True, save=False)
            face_data.append(im)
            break

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        if gui:
            image = QImage(
                frame,
                frame.shape[1],
                frame.shape[0],
                frame.shape[1] * 3,
                QImage.Format_RGB888
            )
            gui.AddPersonLabel.setPixmap(QPixmap.fromImage(image))
        else:
            cv.imshow('frame', frame)

    cv.destroyAllWindows()

    if gui:
        gui.AddPersonLabel.clear()

    return face_data

def take_image(gui=False):
    while True:
        ret, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        image = image_cropping(im=gray, findface=True)

        if gui:
            image_gui = QImage(
                frame,
                frame.shape[1],
                frame.shape[0],
                frame.shape[1] * 3,
                QImage.Format_RGB888
            )
            gui.IdentifySearchLabel.setPixmap(QPixmap.fromImage(image_gui))
        else:
            cv.imshow('frame', frame)

        if image is not False:
            cv.destroyAllWindows()

            break

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    if gui:
        gui.IdentifySearchLabel.clear()

    return image

if __name__ == "__main__":
    # image_selection()
    # image_cropping('s5', '1.pgm')
    # detect_face(r'C:\Users\Aniesia\PycharmProjects\FaceRecognition\Data\\new_faces_notindetected\ja1.jpg')
    face_recording()
    take_image()




