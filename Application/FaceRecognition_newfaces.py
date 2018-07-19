from FaceRecognition_ImagePreprocessing import image_cropping
from auxiliary.aux_plotting import compare_plot

import cv2 as cv
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

class EigenfaceRecognitionNewfaces:

    face_data = None
    newface_dir = None

    def __init__(self, filepath = None, data = None):
        """Either load the data from filepath or get the data directly."""
        if filepath:
            self.face_data = pickle.load(open(filepath, "rb"))
        if data:
            self.face_data = data

        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier.fit(self.face_data.face_weights, self.face_data.labels)
        self.newfacedir = os.path.join(self.face_data.datadir, 'new_faces_notindetected')

    def find_me_face(self, file_path):
        """"Find face most similar to the one in file_path."""

        im = np.array(cv.imread(os.path.join(self.face_data.datadir, file_path), 0), dtype='uint8')
        im = cv.normalize(im, im, 0, 255, cv.NORM_MINMAX)
        face_to_find = self.face_data.transfer_image(im.flatten())

        dist = (self.face_data.face_weights - face_to_find)**2
        dist = np.sqrt(dist.sum(axis=1))
        face_found_id = np.argmin(dist)

        reconstructed_face = np.copy(self.face_data.mean_img)
        reconstructed_face += np.dot(self.face_data.face_weights[face_found_id], self.face_data.eigenfaces_flat)\
            .reshape(self.face_data.image_shape, self.face_data.image_shape)

        compare_plot(im, self.face_data.image_matrix[face_found_id].reshape(86, 86), reconstructed_face)

    def find_me_face_knn(self, file_path):

        im = np.array(cv.imread(os.path.join(self.face_data.datadir, file_path), 0), dtype='uint8')
        im = cv.normalize(im, im, 0, 255, cv.NORM_MINMAX)

        face_to_find = self.face_data.transfer_image(im.flatten())

        if self.is_new_face(face_to_find):

            face_found_id = self.knn_classifier.predict(face_to_find.reshape(1, -1))
            face_found_id = np.where(self.face_data.labels == face_found_id)[0][0]  # it returns a tuple

            reconstructed_face = self.face_data.reconstruct_image(im_id=face_found_id)

            compare_plot(im, self.face_data.image_matrix[face_found_id].reshape(86, 86), reconstructed_face)

        else: print('Could not identify the face')

    def add_face(self, filepath, label=False):
        im = image_cropping(os.path.join(self.newface_dir, filepath), findface=True)

        im_representation = self.face_data.transfer_image(im)
        if self.is_new_face(im_representation, label):



    def update_database(self, image_representation):
        self.face_data.image_matrix.append()

    def is_new_face(self, image_representation):

        probabilities = self.knn_classifier.predict_proba(image_representation.reshape(1, -1))

        dist, ids = self.knn_classifier.kneighbors(X=image_representation.reshape(1, -1),
                                                   n_neighbors=5, return_distance=True)
        person_id = self.face_data.labels[ids]
        print(dist, ids)

        # If too many candidates
        unique_propositions = len(np.unique(person_id))
        if unique_propositions >= 4:
            return False

        # If probability too small
        probability = np.sort(probabilities)[0]
        print(probability[-1], probability[-2])
        if probability[-1] < 0.3:
            return False
        if (probability[-1] - probability[-2]) < 0.1:
            return False

        # If distance between faces of different people is too small
        # First elem is list of unique, second are ids
        face_ids = np.unique(person_id, return_index=True)[1]
        face_distances = [dist.T[x] for x in face_ids]
        # If first closest is far
        if face_distances[0] > 1500:
            return False
        # If first and second are similar
        if face_distances[0] - face_distances[1] > 500:
            return False

        return True


if __name__ == "__main__":
    efr = EigenfaceRecognitionNewfaces(filepath = os.path.join(os.getcwd(), '..', 'Data',
                                                               'Database\\212images-36people.p'))
    # efr.find_me_face('new_faces_notindetected/1.pgm-s5_newface.pgm')
    # efr.find_me_face_knn('new_faces_notindetected/1.pgm-s5_newface.pgm')
    efr.add_face()
