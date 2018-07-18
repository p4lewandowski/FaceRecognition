from FaceRecognition_ImagePreprocessing import image_cropping
import cv2 as cv
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle
from FaceRecognition_eigenfaces import FaceRecognitionEigenfaces

class EigenfaceRecognitionNewfaces:

    face_data = None

    def __init__(self, filepath = None, data = None):
        """Either load the data from filepath or get the data directly."""
        if filepath:
            face_data = pickle.load(open(filepath, "rb"))
        if data:
            face_data = data




    def find_me_face(self, file_path):
        """"Find face most similar to the one in file_path."""

        im = np.array(cv.imread(os.path.join(self.face_data.datadir, file_path), 0), dtype='uint8')
        im = cv.normalize(im, im, 0, 255, cv.NORM_MINMAX)
        im_mean = im.flatten() - self.mean_img.flatten().transpose()

        face_to_find = np.matmul(self.eigenfaces_flat, im_mean)
        dist = (self.face_weights - face_to_find)**2
        dist = np.sqrt(dist.sum(axis = 1))
        face_found_id = np.argmin(dist)

        reconstructed_face = np.copy(self.mean_img)
        reconstructed_face += np.dot(self.face_weights[face_found_id], self.eigenfaces_flat)\
            .reshape(self.image_shape, self.image_shape)

        compare_plot(im, self.image_matrix[face_found_id].reshape(86, 86), reconstructed_face)

    def find_me_face_knn(self, file_path):
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier.fit(self.face_weights, self.labels)

        im = np.array(cv.imread(os.path.join(self.datadir, file_path), 0), dtype='uint8')
        im = cv.normalize(im, im, 0, 255, cv.NORM_MINMAX)
        im_mean = im.flatten() - self.mean_img.flatten().transpose()

        face_to_find = np.matmul(self.eigenfaces_flat, im_mean)
        face_to_find_proba = self.knn_classifier.predict_proba(face_to_find.reshape(1, -1))

        dist, ids = self.knn_classifier.kneighbors(X=face_to_find.reshape(1, -1), n_neighbors=5, return_distance=True)


        if is_new_face(face_to_find_proba):

            face_found_id = self.knn_classifier.predict(face_to_find.reshape(1, -1))
            face_found_id = np.where(self.labels == face_found_id)[0][0]  # it returns a tuple

            reconstructed_face = np.copy(self.mean_img)
            reconstructed_face += np.dot(self.face_weights[face_found_id], self.eigenfaces_flat) \
                .reshape(self.image_shape, self.image_shape)

            compare_plot(im, self.image_matrix[face_found_id].reshape(86, 86), reconstructed_face)

    def add_face(self, filepath):
        im = image_cropping(filepath)
        im = self.transfer_image(im)
        if is_new_face():
            None


def is_new_face(probabilities):

    probability = np.sort(probabilities)[0]
    print(probability[-1], probability[-2])
    if probability[-1] < 0.3:
        return False
    if (probability[-1] - probability[-2]) < 0.1:
        return False
    return True


if __name__ == "__main__":
    efr = EigenfaceRecognitionNewfaces(
        r'C:\Users\Aniesia\PycharmProjects\FaceRecognition\Data\Database\212images-36people.p')
    # image_cropping('s5', '1.pgm')