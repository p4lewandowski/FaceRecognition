from auxiliary.aux_plotting import reconstruction_fast, plot_eigenfaces, \
    reconstruction, plot_faces_2components, compare_plot, reconstruction_manual, \
    plot_tsne
import cv2 as cv
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

class FaceRecognition():

    def get_images(self):
        """Get images and determine mean_img as well as number of images
        and image shape."""

        rootdir = os.getcwd()
        datadir = os.path.join(rootdir, 'detected_faces')
        self.image_matrix = []
        self.eigenfaces_n = 0.99
        self.image_count = 0

        # Go through all the files and read in image, flatten them and append to matrix
        for file in os.listdir(datadir):
            im = cv.imread(os.path.join(datadir, file), 0)

            self.image_matrix.append(np.array(im).flatten())
            self.image_count += 1
        self.image_shape = im.shape[0]

        # Calculate the mean per pixel(feature), normalise it by the amount of pixels
        # and receive 'mean face'
        self.image_matrix_flat = np.array(np.transpose(self.image_matrix))
        self.mean_img = np.sum(self.image_matrix_flat, axis=1) / np.shape(self.image_matrix_flat)[1]
        self.mean_img = self.mean_img.reshape(self.image_shape, self.image_shape)


    def get_eigenfaces(self):
        """ eigenfaces_n is passed as a parameter (percentage of variance to obtain
        and it is changed here to the number of eigenfaces obtained."""

        # Subtract the mean from every flattened image and prepare covariance matrix
        # equal to  L^T*L  for computational efficiency
        self.image_matrix_flat = np.array([x - self.mean_img.flatten() for x in self.image_matrix_flat.transpose()]).transpose()
        cov_matrix = np.matmul(self.image_matrix_flat.transpose(), self.image_matrix_flat)
        cov_matrix /= self.image_count

        # Calculate and choose eigenvectors corresponding to the highest eigenvalues
        pca = PCA(n_components = self.eigenfaces_n)
        pca.fit(cov_matrix)

        # Left multiply to get the correct eigenvectors
        eigenvectors = np.matmul(self.image_matrix_flat, np.transpose(pca.components_))
        pca = PCA(n_components = self.eigenfaces_n)
        pca.fit(eigenvectors.transpose())
        self.eigenfaces_flat = pca.components_
        self.eigenfaces_n = len(pca.components_)


        # Calculate weights representing faces in new dimensional space
        # and reshape eigenface matrix to have
        # number_of_faces X shape1 X shape 2
        self.face_weights = np.matmul(self.image_matrix_flat.transpose(), self.eigenfaces_flat.transpose())
        self.eigenfaces = np.array(self.eigenfaces_flat).reshape((self.eigenfaces_n, self.image_shape, self.image_shape))

    def find_me_face(self, file_path):
        """"Find face most similar to the one in file_path."""

        im = np.array(cv.imread(os.path.join(os.getcwd(), file_path), 0), dtype='float64')
        im_mean = im.flatten() - self.mean_img.flatten().transpose()

        face_to_find = np.matmul(self.eigenfaces_flat, im_mean)
        dist = (self.face_weights - face_to_find)**2
        dist = np.sqrt(dist.sum(axis = 1))
        face_found_id = np.argmin(dist)

        reconstructed_face = np.copy(self.mean_img)
        reconstructed_face += np.dot(self.face_weights[face_found_id], self.eigenfaces_flat)\
            .reshape(self.image_shape, self.image_shape)

        compare_plot(im, self.image_matrix[face_found_id].reshape(86, 86), reconstructed_face)

    def stochastic_neighbour_embedding(self):
        """"Show representation of faces in multidimensional space using t-distributed
         Stochastic Neighbor Embedding."""

        self.t_sne = TSNE(n_components=2, perplexity=5, early_exaggeration=16).fit_transform(self.face_weights)
        plot_tsne(self)


    def show_me_things(self):
        # Plots
        plot_eigenfaces(self)
        plot_faces_2components(self)
        self.stochastic_neighbour_embedding()
        reconstruction(self)
        reconstruction_fast(self)
        reconstruction_manual(self)


