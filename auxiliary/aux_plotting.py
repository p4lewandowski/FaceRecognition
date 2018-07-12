import numpy as np
import matplotlib.pyplot as plt
import time
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import cv2 as cv


rootdir = os.getcwd()
datadir = os.path.join(rootdir, 'detected_faces')

def plot_eigenfaces(images, rows=3, cols=3):

    eigenface_titles = ["eigenface %d" % i for i in range(images.shape[0])]
    plt.figure(figsize=(2 * cols, 2 * rows))
    plt.subplots_adjust(bottom=0.1, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i], cmap=plt.cm.gray)
        plt.title(eigenface_titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('First {} eigenfaces'.format(rows*cols), fontsize = 14)
    plt.show()


def reconstruction(eigenfaces, average_face, face_weights):

    reconstructed_face = average_face
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()

    for id, eigenface in enumerate(eigenfaces):
        reconstructed_face += np.dot(face_weights[55][id], eigenface)
        time.sleep(0.03)
        ax.imshow(reconstructed_face, cmap=plt.cm.bone)
        fig.canvas.draw()
        fig.canvas.flush_events()

    ax.imshow(reconstructed_face, cmap=plt.cm.bone)
    plt.ioff()
    plt.show()

def reconstruction_fast(eigenfaces, average_face, face_weights, id):

    reconstructed_face = average_face
    fig = plt.figure()

    ax = fig.add_subplot(121)
    for iter, eigenface in enumerate(eigenfaces):
        reconstructed_face += np.dot(face_weights[id-100][iter], eigenface)
    ax.imshow(reconstructed_face, cmap=plt.cm.bone)

    ax2 = fig.add_subplot(122)
    for file in os.listdir(datadir):
        if int(file.split('.')[0]) == id:
            img = cv.imread(os.path.join(datadir, '{}.jpg'.format(id)))
            ax2.imshow(img)


    plt.suptitle('Reconstructed face - comparison', fontsize=14)
    plt.show()

def plot_faces_2components(image_matrix_flat, eigenfaces_flat, image_count, image_shape, face_weights):
    face_weights_plot = np.matmul(image_matrix_flat.transpose(), eigenfaces_flat[:2].transpose()).transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    images_to_plot = np.array(np.transpose(image_matrix_flat)).reshape(image_count, image_shape, image_shape)
    plt.imshow(images_to_plot[0])

    for id, coord in enumerate(np.array(face_weights_plot.T)):
        ab = AnnotationBbox(
            OffsetImage(images_to_plot[id], zoom=0.3, cmap=plt.cm.gray),
            coord,
            pad=0,
            xybox=(30., -30.),
            boxcoords="offset points")
        ax.add_artist(ab)

    ax.grid(True)
    ax.set_xlim([np.min(face_weights[:, 0]) * 1.1, np.max(face_weights[:, 0]) * 1.5])
    ax.set_ylim([np.min(face_weights[:, 1]) * 1.5, np.max(face_weights[:, 1]) * 1.2])
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.suptitle('Visualisation of data for first two principal components', fontsize=14)
    plt.draw()
    plt.show()

def compare_plot(im1, im2):
    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(im1, cmap=plt.cm.bone)

    ax2 = fig.add_subplot(122)
    ax2.imshow(im2, cmap=plt.cm.bone)

    plt.suptitle('Image comparison', fontsize=14)
    plt.show()