from FaceRecognition_ImagePreprocessing import image_cropping, face_recording, take_image
from FaceRecognition_eigenfaces import FaceRecognitionEigenfaces
from auxiliary.aux_plotting import compare_plot



import cv2 as cv
import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
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


    def recognize_face(self, image_path=False):
        """Check if face can be assigned to some person in the database with some degree of confidence
        (which is distance of class1 * 1.6 > class2 as we search for the closest class to face).
        The class is chosen based on return from the 'sum_class_distances' function.
        Returns:
            - Find id of the best suited class.
            - Return True if 'class confidence' is fulfilled and False if not."""

        # Get image representation
        if image_path:
            image = cv.imread(image_path, 0)
            image = image_cropping(image)
        else:
            image = take_image()
        image_representation = efr.face_data.transfer_image(image.flatten())

        # fit again in case new data appears
        self.knn_classifier.fit(self.face_data.face_weights, self.face_data.labels)
        probabilities = self.knn_classifier.predict_proba(image_representation.reshape(1, -1))
        prob_person = np.argsort(probabilities)[0][-3:]
        prob_val = probabilities[0][prob_person]

        isnew = 0
        dist, ids = self.knn_classifier.kneighbors(X=image_representation.reshape(1, -1),
                                                   n_neighbors=5, return_distance=True)
        person_ids = self.face_data.labels[ids]
        candidates_n = len([1 for x in prob_val if x > 0])

        class_distances = sum_class_distances(dist, person_ids)
        class_distances = sorted(class_distances, key=lambda x: x[1])

        # If too many candidates
        if candidates_n == 1:
            face_found_id = prob_person[-1]

        else:
            # If too many candidates =  not confident
            if candidates_n >= 4:
                isnew = 1

            # Check for 'confidence'
            # If there is more than one candidate
            elif candidates_n != 1:
                if class_distances[1][1][0] < class_distances[0][1][0] * 1.6:
                    isnew = 1

            # if one class more probable pick this one
            if prob_val[-1] > prob_val[-2]:
                face_found_id = prob_person[-1]
            # If equally probable choose based on distace
            elif prob_val[-1] == prob_val[-2]:
                if class_distances[0][1][0] < class_distances[1][1][0]:
                    face_found_id = prob_person[-1]
                else:
                    face_found_id = prob_person[-2]

        if isnew:
            print("Face found without confidence, label = {}".format(face_found_id))
            return False, face_found_id

        if not isnew:
            print("Face found with confidence, label = {}".format(face_found_id))
            return True, face_found_id


    def add_person(self, **kwargs):
        gui = kwargs.get('gui', False)
        if gui:
            data = face_recording(gui)
        else:
            data = face_recording()
        data = np.array(data)

        label = max(self.face_data.labels) + 1
        count = len(data)
        data = data.reshape(count, self.face_data.image_shape**2)

        for image in data:
            self.add_image2database(image, label)
        print("Added face with label = {}.".format(label))


    def add_image2database(self, image, label):
        """Input is a flattened image."""

        # Increase image count to keep consistency
        self.face_data.image_count += 1
        # Label the image
        self.face_data.labels = np.append(self.face_data.labels, label)

        # Create new 'mean' image
        self.face_data.image_matrix_raw = np.vstack((self.face_data.image_matrix_raw.T, image)).T
        self.mean_img = np.sum(self.face_data.image_matrix_raw, axis=1) / self.face_data.image_count
        self.mean_img = self.mean_img.reshape(self.face_data.image_shape, self.face_data.image_shape)

        # Substract mean from raw images
        self.face_data.image_matrix_flat = np.array(
            [x - self.mean_img.flatten() for x in self.face_data.image_matrix_raw.transpose()]).transpose()

        # Add face weights for the given image
        self.face_data.face_weights = np.matmul(self.face_data.image_matrix_flat.transpose(),
                                                self.face_data.eigenfaces_flat.transpose())


    def update_database(self, image_representation):
        self.face_data.image_matrix.append()


def sum_class_distances(distances, class_labels):
    """Sums distances to face from respective classes and calculates the average."""
    un_val = np.unique(class_labels)
    arr = []
    for i in un_val:
        sum = 0
        count = 0;
        arr_pos = 0;
        for elem in class_labels[0]:
            if elem==i:
                sum +=distances[0][arr_pos]
                count+=1
            arr_pos+=1
        arr.append(np.vstack((i, sum/count)))

    return arr



if __name__ == "__main__":
    fr = FaceRecognitionEigenfaces()
    fr.get_images()
    fr.get_eigenfaces()
    fr.save_to_file()
    efr = EigenfaceRecognitionNewfaces(data=fr)

    efr.add_person()
    efr.recognize_face()
