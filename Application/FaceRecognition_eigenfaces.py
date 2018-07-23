from auxiliary.aux_plotting import reconstruction_fast, plot_eigenfaces, \
    reconstruction, plot_faces_2components, compare_plot, reconstruction_manual, \
    plot_tsne, plot_eigenfaces_variance

import cv2 as cv
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle


class FaceRecognitionEigenfaces():

    rootdir = os.getcwd()
    datadir = os.path.join(rootdir, '..', 'Data')

    def get_images(self):
        """Get images and determine mean_img as well as number of images
        and image shape."""

        self.eigenfaces_n = 0.99
        self.image_count = 0
        self.labels = []
        imagedir = os.path.join(self.datadir, 'detected_faces')


        # Go through all the files and read in image, flatten them and append to matrix
        image_matrix = []
        for file in os.listdir(imagedir):
            im = cv.imread(os.path.join(imagedir, file), 0)

            image_matrix.append(np.array(im).flatten())
            self.image_count += 1
            self.labels.append(int(file.split('_')[1].split('.')[0]))
        self.image_shape = im.shape[0]
        self.labels = np.array(self.labels)

        # Calculate the mean per pixel(feature), normalise it by the amount of pixels
        # and receive 'mean face'
        self.image_matrix_raw = np.array(np.transpose(image_matrix))
        self.mean_img = np.sum(self.image_matrix_raw, axis=1) / self.image_count
        self.mean_img = self.mean_img.reshape(self.image_shape, self.image_shape)
        # Subtract the mean from every flattened image
        self.image_matrix_flat = np.array(
            [x - self.mean_img.flatten() for x in self.image_matrix_raw.transpose()]).transpose()

    def get_eigenfaces(self):
        """ eigenfaces_n is passed as a parameter (percentage of variance to obtain
        and it is changed here to the number of eigenfaces obtained."""

        # Prepare covariance matrix equal to  L^T*L  for computational efficiency
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
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

    def stochastic_neighbour_embedding(self):
        """"Show representation of faces in multidimensional space using t-distributed
         Stochastic Neighbor Embedding."""

        self.t_sne = TSNE(n_components=2, perplexity=5, early_exaggeration=16).fit_transform(self.face_weights)
        plot_tsne(self)

    def transfer_image(self, image):
        """Transfers image to multidimensional representation using eigenfaces
        Input should be flat."""
        image = image - self.mean_img.flatten().transpose()
        image = np.matmul(self.eigenfaces_flat, image)
        return image

    def reconstruct_image(self, im_id = True, weights = False):

        reconstructed_face = np.copy(self.mean_img)

        if im_id:
            reconstructed_face = np.copy(self.mean_img)
            reconstructed_face += np.dot(self.face_weights[im_id], self.eigenfaces_flat) \
                .reshape(self.image_shape, self.image_shape)
        if weights:
            reconstructed_face += np.dot(weights, self.eigenfaces_flat) \
                .reshape(self.image_shape, self.image_shape)

        return reconstructed_face

    def show_me_things(self):
        # Plots
        plot_eigenfaces(self)
        plot_eigenfaces_variance(self)
        plot_faces_2components(self)
        self.stochastic_neighbour_embedding()
        reconstruction(self)
        reconstruction_fast(self)
        reconstruction_manual(self)

    def save_to_file(self):
        dbdir = os.path.join(self.datadir,'Database')
        pickle.dump(self, open("{}\\{}images-{}people.p".format(dbdir, self.image_count,
                                                               len(np.unique(self.labels))), "wb"))








