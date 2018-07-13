from auxiliary.aux_plotting import reconstruction_fast, plot_eigenfaces, \
    reconstruction, plot_faces_2components, compare_plot
import cv2 as cv
import os
import numpy as np
from sklearn.decomposition import PCA
from matplotlib.widgets import Slider, Button, RadioButtons
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


rootdir = os.getcwd()
datadir = os.path.join(rootdir, 'detected_faces')
image_matrix = []
eigenfaces_num = 111
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
print(face_found_id+100)


# Plots
# plot_eigenfaces(eigenfaces)
# plot_faces_2components(image_matrix_flat, eigenfaces_flat, image_count, image_shape, face_weights)
# reconstruction_fast(eigenfaces, mean_img, face_weights, 274)
# compare_plot(im, image_matrix[face_found_id].reshape(86, 86))


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
plt.subplots_adjust(right = 0.65)
ax1.imshow(mean_img, cmap=plt.cm.bone)

ax1_e = plt.axes([0.77, 0.7, 0.15, 0.03])
ax2_e = plt.axes([0.77, 0.65, 0.15, 0.03])
ax3_e = plt.axes([0.77, 0.6, 0.15, 0.03])
ax4_e = plt.axes([0.77, 0.55, 0.15, 0.03])
ax5_e = plt.axes([0.77, 0.5, 0.15, 0.03])
ax6_e = plt.axes([0.77, 0.45, 0.15, 0.03])
ax7_e = plt.axes([0.77, 0.4, 0.15, 0.03])
ax8_e = plt.axes([0.77, 0.35, 0.15, 0.03])
ax9_e = plt.axes([0.77, 0.3, 0.15, 0.03])


# mean face to weights
mean_face_9pca = np.matmul(face_weights[0], eigenfaces_flat)[:9]

# Back
reconstructed_face_template = mean_img
for iter, eigenface in enumerate(eigenfaces[:9]):
    reconstructed_face_template += np.dot(mean_face_9pca[iter], eigenface)
ax2.imshow(reconstructed_face_template, cmap=plt.cm.bone)


slider_1 = Slider(ax1_e, 'weight 1', -5000, 5000, valinit=mean_face_9pca[0], valfmt="%.0f")
slider_2 = Slider(ax2_e, 'weight 2', -5000, 5000, valinit=mean_face_9pca[1], valfmt="%.0f")
slider_3 = Slider(ax3_e, 'weight 3', -5000, 5000, valinit=mean_face_9pca[2], valfmt="%.0f")
slider_4 = Slider(ax4_e, 'weight 4', -5000, 5000, valinit=mean_face_9pca[3], valfmt="%.0f")
slider_5 = Slider(ax5_e, 'weight 5', -5000, 5000, valinit=mean_face_9pca[4], valfmt="%.0f")
slider_6 = Slider(ax6_e, 'weight 6', -5000, 5000, valinit=mean_face_9pca[5], valfmt="%.0f")
slider_7 = Slider(ax7_e, 'weight 7', -5000, 5000, valinit=mean_face_9pca[6], valfmt="%.0f")
slider_8 = Slider(ax8_e, 'weight 8', -5000, 5000, valinit=mean_face_9pca[7], valfmt="%.0f")
slider_9 = Slider(ax9_e, 'weight 9', -5000, 5000, valinit=mean_face_9pca[8], valfmt="%.0f")

def update(val):
    mean_face_9pca = [slider_1.val, slider_2.val, slider_3.val, slider_4.val,
                      slider_5.val, slider_6.val, slider_7.val, slider_8.val,
                      slider_9.val]

    reconstructed_face = np.copy(mean_img)
    for iter, eigenface in enumerate(eigenfaces[:9]):
        reconstructed_face += np.dot(mean_face_9pca[iter], eigenface)

    ax2.imshow(reconstructed_face, cmap=plt.cm.bone)

slider_1.on_changed(update);slider_2.on_changed(update);slider_3.on_changed(update)
slider_4.on_changed(update);slider_5.on_changed(update);slider_6.on_changed(update)
slider_7.on_changed(update);slider_8.on_changed(update);slider_9.on_changed(update)


def reset(event):
    slider_1.reset(); slider_2.reset(); slider_3.reset(); slider_4.reset();
    slider_5.reset(); slider_6.reset(); slider_7.reset(); slider_8.reset();
    slider_9.reset();
    ax2.imshow(reconstructed_face_template, cmap=plt.cm.bone)


resetax = plt.axes([0.45, 0.05, 0.1, 0.1])
button = Button(resetax, 'Reset', hovercolor='0.975')
button.on_clicked(reset)

plt.show()

