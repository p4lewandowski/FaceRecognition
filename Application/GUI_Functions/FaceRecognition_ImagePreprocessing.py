import cv2 as cv
import os
from matplotlib.image import imsave
import matplotlib.pyplot as plt

scale_factor = 1.15
min_neighbors = 3


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

def image_cropping(im = None, filepath = None, findface = False, save=True):
    """Read in the image either from file or from image frame from a camera.
    Either save it to file or just return the image."""
    if filepath:
        im = cv.imread(filepath, 0)

    if findface:
        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        face = face_cascade.detectMultiScale(im, scale_factor, min_neighbors)
        for (x, y, w, h) in face:
            crop_img = im[y:y + h, x:x + w]

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
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(im, scale_factor, min_neighbors)
    for (x, y, w, h) in faces:
        cv.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv.imshow('img', im)
    cv.waitKey(0)
    cv.destroyAllWindows()

def face_recording():
    cap = cv.VideoCapture(0)
    cap.set(3, 640)  # WIDTH
    cap.set(4, 480)  # HEIGHT

    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    face_data = []

    while (True):
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
            # im = image_cropping(im=gray, findface=True, save=False)
            # face_data.append(im)
            break

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    return face_data


if __name__ == "__main__":
    # image_selection()
    # image_cropping('s5', '1.pgm')
    # detect_face(r'C:\Users\Aniesia\PycharmProjects\FaceRecognition\Data\\new_faces_notindetected\ja1.jpg')
    face_recording()




