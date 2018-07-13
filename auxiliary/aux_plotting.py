import numpy as np
import matplotlib.pyplot as plt
import time
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2 as cv
from matplotlib.widgets import Slider, Button, RadioButtons


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

    reconstructed_face = np.copy(average_face)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()

    for id, eigenface in enumerate(eigenfaces):
        reconstructed_face += np.dot(face_weights[55][id], eigenface)
        #time.sleep(0.01)
        print(id)
        ax.imshow(reconstructed_face, cmap=plt.cm.bone)
        fig.canvas.draw()
        fig.canvas.flush_events()

    ax.imshow(reconstructed_face, cmap=plt.cm.bone)
    plt.ioff()
    plt.show()

def reconstruction_fast(eigenfaces, average_face, face_weights, id):

    reconstructed_face = np.copy(average_face)
    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    for iter, eigenface in enumerate(eigenfaces):
        reconstructed_face += np.dot(face_weights[id-100][iter], eigenface)
    ax2.imshow(reconstructed_face, cmap=plt.cm.bone)

    for file in os.listdir(datadir):
        if int(file.split('.')[0]) == id:
            img = cv.imread(os.path.join(datadir, '{}.jpg'.format(id)))
            ax.imshow(img)


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

def reconstruction_manual(mean_img, eigenfaces, face_weights):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    plt.subplots_adjust(right=0.65)
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
    mean_face_9pca = np.matmul(face_weights[0], eigenfaces.reshape(np.shape(eigenfaces)[0], 86*86))[:9]

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

    slider_1.on_changed(update);
    slider_2.on_changed(update);
    slider_3.on_changed(update)
    slider_4.on_changed(update);
    slider_5.on_changed(update);
    slider_6.on_changed(update)
    slider_7.on_changed(update);
    slider_8.on_changed(update);
    slider_9.on_changed(update)

    def reset(event):
        slider_1.reset();
        slider_2.reset();
        slider_3.reset();
        slider_4.reset();
        slider_5.reset();
        slider_6.reset();
        slider_7.reset();
        slider_8.reset();
        slider_9.reset();
        ax2.imshow(reconstructed_face_template, cmap=plt.cm.bone)

    resetax = plt.axes([0.45, 0.05, 0.1, 0.1])
    button = Button(resetax, 'Reset', hovercolor='0.975')
    button.on_clicked(reset)

    plt.show()
