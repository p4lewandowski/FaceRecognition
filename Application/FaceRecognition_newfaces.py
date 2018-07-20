from FaceRecognition_ImagePreprocessing import image_cropping
from auxiliary.aux_plotting import compare_plot

import cv2 as cv
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

class EigenfaceRecognitionNewfaces:

    face_data = None

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

        compare_plot(im, self.face_data.image_matrix_flat.T[face_found_id].reshape(86, 86), reconstructed_face)

    def find_me_face_knn(self, file_path):

        im = np.array(cv.imread(os.path.join(self.face_data.datadir, file_path), 0), dtype='uint8')
        im = cv.normalize(im, im, 0, 255, cv.NORM_MINMAX)

        face_to_find = self.face_data.transfer_image(im.flatten())

        face_recognized, found_id = self.recognize_face(face_to_find)

        if face_recognized:

            reconstructed_face = self.face_data.reconstruct_image(im_id=found_id)
            compare_plot(im, self.face_data.image_matrix_flat.T[found_id].reshape(86, 86), reconstructed_face)

        else: print('Could not identify the face - hard to distinguish')

    def add_face(self, filepath, label=False):
        im = image_cropping(os.path.join(self.newfacedir, filepath), findface=True)
        im_flat = im.flatten()
        im_representation = self.face_data.transfer_image(im_flat)
        face_recognized, found_id = self.recognize_face(im_representation)
        im_mean = im - self.face_data.mean_img

        if not face_recognized:
            print("New face detected or classified improperly")

        self.face_data.image_matrix_flat = self.face_data.image_matrix_flat.T
        self.face_data.image_matrix_flat = np.vstack((self.face_data.image_matrix_flat, im_mean.flatten()))
        self.face_data.image_matrix_flat = self.face_data.image_matrix_flat.T
        self.face_data.face_weights = np.matmul(
            self.face_data.image_matrix_flat.transpose(), self.face_data.eigenfaces_flat.transpose())

        self.face_data.image_count += 1

        if label:
            self.face_data.labels = np.append(self.face_data.labels, label)
        else:
            self.face_data.labels = np.append(self.face_data.labels, found_id)


    def update_database(self, image_representation):
        self.face_data.image_matrix.append()

    def recognize_face(self, image_representation):
        """Check if face can be assigned to some person in the database.
        If it can, find id of this person. If it cannot, find free id."""

        # fit again in case new data appears
        self.knn_classifier.fit(self.face_data.face_weights, self.face_data.labels)
        # probabilities = self.knn_classifier.predict_proba(image_representation.reshape(1, -1))
        # probability = np.sort(probabilities)[0]

        isnew = 0
        dist, ids = self.knn_classifier.kneighbors(X=image_representation.reshape(1, -1),
                                                   n_neighbors=5, return_distance=True)
        person_id = self.face_data.labels[ids]

        # First elem is list of unique, second are ids
        face_ids = np.unique(person_id, return_index=True)[1]
        # Distances from different people
        face_distances = [dist.T[x] for x in face_ids]
        face_distances = face_distances[::-1]

        print('###', dist, person_id[0])

        # If too many candidates
        unique_propositions = len(np.unique(person_id))
        if unique_propositions >= 4:
            isnew = 1

        # If distance between faces of different people is too small
        elif len(face_distances)>1:
            if abs(face_distances[0] - face_distances[1]) < 700:
                isnew = 1

        if not isnew:
            face_found_id = self.knn_classifier.predict(image_representation.reshape(1, -1))
            face_found_id = np.where(self.face_data.labels == face_found_id)[0][0]  # it returns a tuple

            return True, face_found_id

        return False, max(self.face_data.labels)+1


if __name__ == "__main__":
    efr = EigenfaceRecognitionNewfaces(filepath = os.path.join(os.getcwd(), '..', 'Data',
                                                               'Database\\212images-36people.p'))
    # efr.find_me_face('new_faces_notindetected/1.pgm-s5_newface.pgm')
    # efr.find_me_face_knn('new_faces_notindetected/1.pgm-s5_newface.pgm')
    efr.add_face('ja1.jpg', 999)
    efr.add_face('ja2.jpg', 999)
    efr.add_face('ja3.jpg', 999)
    efr.add_face('ja4.jpg', 999)
    efr.add_face('ja5.jpg', 999)
    efr.add_face('ja6.jpg')
    efr.add_face('ja11.jpg')

    efr.face_data.stochastic_neighbour_embedding()
    efr.face_data.show_me_things()

    #Jak dodawac ludzi, z jakim label, kiedy ich klasyfikowac skoro knn ma 5 neighourow
    # jak to mialoby wygladac z kamera
    # czy zmieniac wtedy twarz srednia? Twarze wlasne?