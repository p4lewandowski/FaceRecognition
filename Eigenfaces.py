import cv2 as cv
import os
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time

rootdir = 'C:\\Users\\Aniesia\\PycharmProjects\\FaceRecognition\\'
datadir = os.path.join(rootdir, 'detected_faces')
image_matrix = []
eigenfaces_num = 50

def plot_gallery(images, titles, h, w, n_row=4, n_col=2):
    """Helper function to plot a gallery of portraits"""

    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

def reconstruction(eigenfaces, average_face):

    reconstructed_face = average_face
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()

    for id, eigenface in enumerate(eigenfaces):
        reconstructed_face += np.dot(face_weigths[5][id], eigenface)
        time.sleep(0.1)
        ax.imshow(reconstructed_face, cmap=plt.cm.bone)
        fig.canvas.draw()
        fig.canvas.flush_events()

    plt.ioff()
    plt.show()



for file in os.listdir(datadir):
    im = cv.imread(os.path.join(datadir, file), 0)
    image_shape = im.shape
    im = np.array(im).reshape(86*86)
    image_matrix.append(im)


image_matrix = np.array(image_matrix)

pca = PCA(n_components= eigenfaces_num)
pca.fit(image_matrix)

eigenfaces = pca.components_
eigenfaces = pca.components_.reshape((eigenfaces_num, image_shape[0], image_shape[1]))
average_face = pca.mean_.reshape(image_shape)
face_weigths = pca.transform(image_matrix - pca.mean_)
reconstruction(eigenfaces, average_face)


### Plots ###
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, image_shape[0], image_shape[1])
plt.show()

plt.imshow(average_face, cmap=plt.cm.bone)
plt.show()
