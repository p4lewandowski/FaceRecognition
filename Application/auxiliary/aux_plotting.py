import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import cv2 as cv
from matplotlib.widgets import Slider, Button
import seaborn as sns

rootdir = os.getcwd()
datadir = os.path.join(rootdir, 'detected_faces')

def plot_eigenfaces(fr, rows=3, cols=3):

    eigenface_titles = ["eigenface %d" % i for i in range(1, fr.eigenfaces.shape[0]+1)]
    plt.figure(figsize=(2 * cols, 2 * rows))
    plt.subplots_adjust(bottom=0.1, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fr.eigenfaces[i], cmap=plt.cm.gray)
        plt.title(eigenface_titles[i], size=12)
        plt.xticks(())
        plt.yticks(())
    plt.suptitle('First {} eigenfaces'.format(rows*cols), fontsize = 14)
    plt.show()

def plot_eigenfaces_variance(fr):
    X = list(range(1, fr.eigenfaces_n+1))
    plot = sns.barplot(x=X, y=fr.explained_variance_ratio_)
    plot.set_xlabel('N-ty główny komponent')
    plot.set_ylabel('Wartosc procentowa reprezentowanej wariancji')
    plot.set_title('Procentowa reprezentacja wariancji dla poszczególnych głównych składowych')
    plt.show()

def reconstruction(fr, id = 55):
    """Reconstruct example of a face given by id, from 1 up to im_count"""

    reconstructed_face = np.copy(fr.mean_img)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()
    plt.show()
    plt.axis('off')

    for index, eigenface in enumerate(fr.eigenfaces):
        reconstructed_face += np.dot(fr.face_weights[id][index], eigenface)
        ax.imshow(reconstructed_face, cmap=plt.cm.bone)
        fig.canvas.draw()
        fig.canvas.flush_events()

    ax.imshow(reconstructed_face, cmap=plt.cm.bone)
    plt.ioff()
    plt.close()

def reconstruction_fast(fr, id = 5):
    """Same as reconstruction but instantenous. Additionally, comparison with a real image.
    +100 is added to id as ImagePreprocessing names image files in such a manner."""

    reconstructed_face = np.copy(fr.mean_img)
    fig = plt.figure()
    detected_dir = os.path.join(fr.datadir, 'detected_faces')

    ax = fig.add_subplot(121)
    plt.axis('off')
    plt.title('Real Image')
    ax2 = fig.add_subplot(122)
    plt.axis('off')
    plt.title('Reconstructed Image')

    reconstructed_face += np.dot(fr.face_weights[id], fr.eigenfaces_flat).reshape(fr.image_shape, fr.image_shape)
    ax2.imshow(reconstructed_face, cmap=plt.cm.bone)

    for file in os.listdir(detected_dir):
        if int(file.split('_')[0]) == id+100:
            img = cv.imread(os.path.join(detected_dir, file), 0)
            ax.imshow(img, cmap=plt.cm.bone)


    plt.suptitle('Reconstructed face - comparison', fontsize=14)
    plt.show()

def plot_faces_2components(fr):
    face_weights_plot = np.matmul(fr.image_matrix_flat.transpose(), fr.eigenfaces_flat[:2].transpose()).transpose()
    fig = plt.figure()
    ax = fig.add_subplot(111)

    images_to_plot = np.array(np.transpose(fr.image_matrix_flat))\
        .reshape(fr.image_count, fr.image_shape, fr.image_shape)

    for id, coord in enumerate(np.array(face_weights_plot.T)):
        ab = AnnotationBbox(
            OffsetImage(images_to_plot[id], zoom=0.3, cmap=plt.cm.bone),
            coord,
            pad=0,
            xybox=(0., 0.),
            boxcoords="offset points")
        ax.add_artist(ab)

    ax.grid(True)
    ax.set_xlim([np.min(fr.face_weights[:, 0]) * 1.2, np.max(fr.face_weights[:, 0]) * 1.2])
    ax.set_ylim([np.min(fr.face_weights[:, 1]) * 1.2, np.max(fr.face_weights[:, 1]) * 1.2])
    plt.xlabel('First principal component')
    plt.ylabel('Second principal component')
    plt.suptitle('Visualisation of data for first two principal components', fontsize=14)
    plt.draw()
    plt.show()

def compare_plot(im1, im2, im3):
    fig = plt.figure(figsize = (5, 5))
    #plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False, labelleft=False)

    ax = fig.add_subplot(221)
    plt.axis('off')
    ax.imshow(im1, cmap=plt.cm.bone)
    plt.title("Searched face")

    ax2 = fig.add_subplot(222)
    plt.axis('off')
    ax2.imshow(im2, cmap=plt.cm.bone)
    plt.title("Found face as an image")

    ax3 = fig.add_subplot(212)
    plt.axis('off')
    ax3.imshow(im3, cmap=plt.cm.bone)
    plt.title("Reconstructed found face")

    plt.suptitle('Image comparison', fontsize=14)

    plt.show()

def reconstruction_manual(fr):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
    plt.subplots_adjust(right=0.65)
    ax1.imshow(fr.mean_img, cmap=plt.cm.bone)
    plt.axis('off')

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
    mean_face_9pca = np.matmul(fr.face_weights[0], fr.eigenfaces.reshape(np.shape(fr.eigenfaces)[0], 86*86))[:9]

    # Back
    reconstructed_face_template = fr.mean_img
    for iter, eigenface in enumerate(fr.eigenfaces[:9]):
        reconstructed_face_template += np.dot(mean_face_9pca[iter], eigenface)
    ax2.imshow(reconstructed_face_template, cmap=plt.cm.bone)
    plt.axis('off')

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

        reconstructed_face = np.copy(fr.mean_img)
        for iter, eigenface in enumerate(fr.eigenfaces[:9]):
            reconstructed_face += np.dot(mean_face_9pca[iter], eigenface)

        ax2.imshow(reconstructed_face, cmap=plt.cm.bone)

    slider_1.on_changed(update)
    slider_2.on_changed(update)
    slider_3.on_changed(update)
    slider_4.on_changed(update)
    slider_5.on_changed(update)
    slider_6.on_changed(update)
    slider_7.on_changed(update)
    slider_8.on_changed(update)
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


def plot_tsne(fr):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('TSNE plot')
    ax.scatter(fr.t_sne[:, 0], fr.t_sne[:, 1])

    images_to_plot = np.array(np.transpose(fr.image_matrix_flat))\
        .reshape(fr.image_count, fr.image_shape, fr.image_shape)

    for id, coord in enumerate(np.array(fr.t_sne)):
        ab = AnnotationBbox(
            OffsetImage(images_to_plot[id], zoom=0.3, cmap=plt.cm.bone),
            coord,
            pad=0,
            xybox=(0., 0.),
            boxcoords="offset points")
        ax.add_artist(ab)

    # ab = AnnotationBbox(
    #     OffsetImage(images_to_plot[len(fr.t_sne)-1], zoom=0.3),
    #     fr.t_sne[-1],
    #     pad=0,
    #     xybox=(0., 0.),
    #     boxcoords="offset points")
    # ax.add_artist(ab)

    plt.show()
