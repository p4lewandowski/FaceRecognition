from FaceRecognition_ImagePreprocessing import image_cropping, face_recording
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

    def add__new_face(self, filepath, label=False):
        im = image_cropping(filepath=os.path.join(self.newfacedir, filepath), findface=False) #!!!
        im_flat = im.flatten()
        im_representation = self.face_data.transfer_image(im_flat)
        face_recognized, found_id = self.recognize_face(im_representation)

        if not face_recognized:
            print("New face detected or classified improperly")
        if face_recognized:
            print("Face was detected. Person in the database.")

        if label:
            self.add_image2database(im_flat, label)
        else:
            self.add_image2database(im_flat, found_id)

    def recognize_face(self, image_representation):
        """Check if face can be assigned to some person in the database.
        If it can, find id of this person. If it cannot, find free id."""

        # fit again in case new data appears
        self.knn_classifier.fit(self.face_data.face_weights, self.face_data.labels)
        probabilities = self.knn_classifier.predict_proba(image_representation.reshape(1, -1))
        prob_person = np.argsort(probabilities)[0][-3:]
        prob_val = probabilities[0][prob_person]
        prob_person +=1 # Compensating for the fact that ids begin at 0

        isnew = 0
        dist, ids = self.knn_classifier.kneighbors(X=image_representation.reshape(1, -1),
                                                   n_neighbors=5, return_distance=True)
        person_id = self.face_data.labels[ids]

        # First elem is list of unique, second are ids
        face_ids = np.unique(person_id, return_index=True  )[1]
        # Distances from different people
        face_distances = [dist.T[x] for x in face_ids]
        face_distances = face_distances[::-1]

        class_distances = sum_class_distances(dist, person_id)
        class_distances = sorted(class_distances, key=lambda x: x[1])

        # If too many candidates
        unique_propositions = len(np.unique(person_id))
        if unique_propositions >= 4:
            isnew = 1

        # If average distance between candidate classes is too small
        elif len(face_distances)>1:
            if class_distances[1][1][0] < class_distances[0][1][0] * 1.6:
                isnew = 1

        if prob_val[-1] > prob_val[-2]:
            face_found_id = prob_person[-1]
        elif prob_val[-1] == prob_val[-2]:
            if class_distances[0][1][0]< class_distances[1][1][0]:
                face_found_id = prob_person[-1]
            else:
                face_found_id = prob_person[-2]

        # Nearest neighbout
        return person_id[0]

        if isnew:
            return False, face_found_id
            # return False, max(self.face_data.labels) + 1

        if not isnew:
            return True, face_found_id


    def add_person(self):
        data = face_recording()
        data = np.array(data)

        label = max(self.face_data.labels) + 1
        count = len(data)
        data = data.reshape(count, self.face_data.image_shape**2)

        for image in data:
            # plt.imshow(image.reshape(86, 86), cmap=plt.cm.bone)
            # plt.show()
            # cv.imwrite('a.pgm', image.reshape(self.face_data.image_shape, self.face_data.image_shape))
            self.add_image2database(image, label)


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
    efr = EigenfaceRecognitionNewfaces(filepath = os.path.join(os.getcwd(), '..', 'Data',
                                                               'Database\\212images-36people.p'))
#
#     # fr = FaceRecognitionEigenfaces()
#     # fr.get_images()
#     # fr.get_eigenfaces()
#     # fr.save_to_file()
#     # fr.show_me_things()
#     # efr = EigenfaceRecognitionNewfaces(data=fr)
#
#     # efr.find_me_face('new_faces_notindetected/1.pgm-s5_newface.pgm')
#     # efr.find_me_face_knn('new_faces_notindetected/1.pgm-s5_newface.pgm')
    efr.add__new_face('ja1.jpg', 999)
    efr.add__new_face('ja2.jpg', 999)
    efr.add__new_face('ja3.jpg', 999)
    efr.add__new_face('ja4.jpg', 999)
#     # efr.add__new_face('ja5.jpg', 999)
#     # efr.add__new_face('ja6.jpg')
#     # efr.add__new_face('ja11.jpg')
#
#     efr.add_person()
#     efr.add__new_face('a.pgm')
#     efr.face_data.get_eigenfaces()
#     # efr.face_data.stochastic_neighbour_embedding()
#     efr.face_data.show_me_things()
#
#
#     # fr = FaceRecognitionEigenfaces()
#     # fr.get_images()
#     # fr.get_eigenfaces()
#     # fr.show_me_things()
#     # efr = EigenfaceRecognitionNewfaces(data=fr)
#     # efr.face_data.stochastic_neighbour_embedding()




    # sprawdzic celnosc
    # FAR FRR/True match rates

    # a jak bedzie czas, to czy da sie opisac obraz za pomoca waveletow,
    # falki gabora np. jak zmienic na taka reprezentacje
    # pamietac o tym skalowaniu
    # niech wymiary zostana