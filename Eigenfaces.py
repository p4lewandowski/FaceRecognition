from auxiliary.aux_plotting import reconstruction_fast, plot_eigenfaces, \
    reconstruction, plot_faces_2components, compare_plot, reconstruction_manual
import cv2 as cv
import os
import numpy as np
from sklearn.decomposition import PCA


rootdir = os.getcwd()
datadir = os.path.join(rootdir, 'detected_faces')
image_matrix = []
eigenfaces_num = 30
image_count = 0

# Go through all the files and read in image, flatten them and append to matrix
for file in os.listdir(datadir):
    im = cv.imread(os.path.join(datadir, file), 0)
    image_shape = im.shape[0]
    image_matrix.append(np.array(im).flatten())
    image_count +=1


# Calculate the mean per pixel(feature), normalise it by the amount of pixels
# and receive 'mean face'
image_matrix_flat = np.array(np.transpose(image_matrix))
mean_img = np.sum(image_matrix_flat, axis=1) / np.shape(image_matrix_flat)[1]
mean_img = mean_img.reshape(image_shape, image_shape)


# Subtract the mean from every flattened image and prepare covariance matrix
# equal to  L^T*L  for computational efficiency
image_matrix_flat = np.array([x - mean_img.flatten() for x in image_matrix_flat.transpose()]).transpose()
cov_matrix = np.matmul(image_matrix_flat.transpose(), image_matrix_flat)
cov_matrix /= image_count


# Calculate and choose eigenvectors corresponding to the highest eigenvalues
pca = PCA(n_components = eigenfaces_num)
pca.fit(cov_matrix)

# Left multiply to get the correct eigenvectors
eigenvectors = np.matmul(image_matrix_flat, np.transpose(pca.components_))
pca = PCA(n_components = eigenfaces_num)
pca.fit(eigenvectors.transpose())
eigenfaces_flat = pca.components_


# Calculate weights representing faces in new dimensional space
# and reshape eigenface matrix to have
# number_of_faces X shape1 X shape 2
face_weights = np.matmul(image_matrix_flat.transpose(), eigenfaces_flat.transpose())
eigenfaces = np.array(eigenfaces_flat).reshape((eigenfaces_num, image_shape, image_shape))


# Find Face
im = np.array(cv.imread(os.path.join(rootdir, '180.jpg'), 0), dtype='float64')
im_mean = im.flatten() - mean_img.flatten().transpose()

face_to_find = np.matmul(eigenfaces_flat, im_mean)
dist = (face_weights - face_to_find)**2
dist = np.sqrt(dist.sum(axis = 1))
face_found_id = np.argmin(dist)


# Plots
plot_eigenfaces(eigenfaces)
plot_faces_2components(image_matrix_flat, eigenfaces_flat, image_count, image_shape, face_weights)
reconstruction(eigenfaces, mean_img, face_weights)
reconstruction_fast(eigenfaces, mean_img, face_weights, 274)
compare_plot(im, image_matrix[face_found_id].reshape(86, 86))
reconstruction_manual(mean_img, eigenfaces, face_weights)






# 10 klas sporo reprezenttntow, wyznacz srednia, minimalna odleglosc twarzy od swojego modelu
# zbior 1 zbior 2 i zobaczyc, gdzie jest polowqa
# jesli ja pasuje do jakiejs klasy, to do kazdej powinno mi byc dalej, ile razy gorsze sa inne klasy
# metoda sift

# 22-28.07
# 1-20.08