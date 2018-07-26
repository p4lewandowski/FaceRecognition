import sys
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QDialog
from gui import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from FaceRecognition_eigenfaces import FaceRecognitionEigenfaces
from FaceRecognition_newfaces import EigenfaceRecognitionNewfaces
from FaceRecognition_ImagePreprocessing import image_selection
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import Slider, Button
import cv2 as cv
import os
from matplotlib.image import imsave
import matplotlib.pyplot as plt


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        # Enable access to inherieted variables, methods, classes etc.
        super(self.__class__, self).__init__() # Inheritance

        ################## Initialize the application ##################
        self.setupUi(self)

        ################## Cosmetic settings ##################
        # Setting up icon and window title
        self.setWindowTitle('Basic Face Recognition System')
        self.setWindowIcon(QIcon('app_icon.jpg'))

        # ############### Plot embedded display ################
        self.LearnEigenfaces.clicked.connect(self.DatabaseEigenfaces)


        self.show()
        self.face_recording()

    def face_recording(self):
        scale_factor = 1.15
        min_neighbors = 3
        cap = cv.VideoCapture(0)
        cap.set(3, 640)  # WIDTH
        cap.set(4, 480)  # HEIGHT

        face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
        face_data = []

        while (True):
            # Capture frame-by-frame
            ret, frame = cap.read()
            # Our operations on the frame come here
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            # Display the resulting frame
            for (x, y, w, h) in faces:
                cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]
                im = image_cropping(im=gray, findface=True, save=False)
                face_data.append(im)
                break

            image = QImage(
                frame,
                frame.shape[1],
                frame.shape[0],
                frame.shape[1] * 3,
                QImage.Format_RGB888
            )
            self.AddPersonLabel.setPixmap(QPixmap.fromImage(image))

        # When everything done, release the capture
        cap.release()
        cv.destroyAllWindows()

        return face_data


    def DatabaseEigenfaces(self):
        self.fr = FaceRecognitionEigenfaces()
        self.fr.get_images()
        self.fr.get_eigenfaces()
        self.fr.stochastic_neighbour_embedding()

        self.plotPCAcomponents()
        self.plotEigenfaces()
        self.plotPCA2components()
        self.plotTSNE()
        self.plotReconstructionManual()

    def plotReconstructionManual(self):
        figure = self.create_plots(self.VisualizationReconstructionTab)
        ax1 = figure.add_subplot(121)
        ax2 = figure.add_subplot(122)
        plt.subplots_adjust(right=0.65)
        ax1.imshow(self.fr.mean_img, cmap=plt.cm.bone)
        plt.axis('off')
        fig_title = 'Manualna rekonstrukcja twarzy na podstawie wag i twarzy własnych.'
        plt.text(1.5, 1.45, fig_title,
                 horizontalalignment='center',
                 fontsize=14,
                 transform=ax1.transAxes)

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
        mean_face_9pca = np.matmul(
            self.fr.face_weights[0], self.fr.eigenfaces.reshape(np.shape(self.fr.eigenfaces)[0], 86 * 86))[:9]

        # Back
        reconstructed_face_template = self.fr.mean_img
        for iter, eigenface in enumerate(self.fr.eigenfaces[:9]):
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

            reconstructed_face = np.copy(self.fr.mean_img)
            for iter, eigenface in enumerate(self.fr.eigenfaces[:9]):
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
        self._rec_man_button = Button(resetax, 'Reset', hovercolor='0.8')
        self._rec_man_button.on_clicked(reset)

    def plotTSNE(self):
        figure = self.create_plots(self.VisualizationTSNETab)
        ax = figure.add_subplot(111)
        plt.title('TSNE plot')
        ax.scatter(self.fr.t_sne[:, 0], self.fr.t_sne[:, 1])

        images_to_plot = np.array(np.transpose(self.fr.image_matrix_flat)) \
            .reshape(self.fr.image_count, self.fr.image_shape, self.fr.image_shape)

        for id, coord in enumerate(np.array(self.fr.t_sne)):
            ab = AnnotationBbox(
                OffsetImage(images_to_plot[id], zoom=0.3, cmap=plt.cm.bone),
                coord,
                pad=0,
                xybox=(0., 0.),
                boxcoords="offset points")
            ax.add_artist(ab)
        ax.set_title('Rzutowanie danych do reprezentacji 2D z użyciem TSNE')

    def plotPCA2components(self):
        figure = self.create_plots(self.Visualization2PCATab)
        face_weights_plot = np.matmul(self.fr.image_matrix_flat.transpose(),
                                      self.fr.eigenfaces_flat[:2].transpose()).transpose()
        ax = figure.add_subplot(111)
        images_to_plot = np.array(np.transpose(self.fr.image_matrix_flat))\
            .reshape(self.fr.image_count, self.fr.image_shape, self.fr.image_shape)

        for id, coord in enumerate(np.array(face_weights_plot.T)):
            ab = AnnotationBbox(
                OffsetImage(images_to_plot[id], zoom=0.3, cmap=plt.cm.bone),
                coord,
                pad=0,
                xybox=(0., 0.),
                boxcoords="offset points")
            ax.add_artist(ab)

        ax.grid(True)
        ax.set_xlim([np.min(self.fr.face_weights[:, 0]) * 1.2, np.max(self.fr.face_weights[:, 0]) * 1.2])
        ax.set_ylim([np.min(self.fr.face_weights[:, 1]) * 1.2, np.max(self.fr.face_weights[:, 1]) * 1.2])
        ax.set_xlabel('First principal component')
        ax.set_ylabel('Second principal component')
        ax.set_title('Visualisation of data for first two principal components')



    def plotEigenfaces(self):
        figure = self.create_plots(self.VisualizationEigenfacesTab, toolbar=False)
        rows = 3
        cols = 3
        eigenface_titles = ["eigenface %d" % i for i in range(1, self.fr.eigenfaces.shape[0] + 1)]
        figure.subplots_adjust(bottom=0.1, left=.01, right=.99, top=.90, hspace=.35)
        for i in range(rows * cols):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(self.fr.eigenfaces[i], cmap=plt.cm.gray)
            plt.title(eigenface_titles[i], size=12)
            plt.xticks(())
            plt.yticks(())
        figure.suptitle('First {} eigenfaces'.format(rows * cols), fontsize=14)

    def plotPCAcomponents(self):
        figure = self.create_plots(self.VisualizationPCACoeffTab)
        ax = figure.add_subplot(111)
        X = list(range(1, self.fr.eigenfaces_n + 1))
        ax.bar(X, self.fr.explained_variance_ratio_)
        ax.set_xlabel('N-ty główny komponent')
        ax.set_ylabel('Wartosc procentowa reprezentowanej wariancji')
        ax.set_title('Procentowa reprezentacja wariancji dla poszczególnych głównych składowych')

    def create_plots(self, parent, toolbar = True):
        figure = plt.figure(figsize=(7.7, 5))
        figure.subplots_adjust(top=.85)
        canvas = FigureCanvas(figure)
        canvas.setParent(parent)
        if toolbar:
            toolbar = NavigationToolbar(canvas, self)
            toolbar.setParent(parent)
        figure.clear()
        return figure

if __name__ == '__main__':        # if we're running file directly and not importing it
    app = QApplication(sys.argv)  # A new instance of QApplication
    form = MainWindow()  # New instance of application
    form.show()  # Show the form
    app.exec_()  # and execute the app