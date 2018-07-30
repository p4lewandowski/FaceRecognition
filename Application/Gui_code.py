import sys

from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import cv2 as cv
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.widgets import Slider, Button
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox

from GUI_Components.gui import Ui_MainWindow
from FaceRecognition_eigenfaces import FaceRecognitionEigenfaces
from FaceRecognition_newfaces import EigenfaceRecognitionNewfaces
from FaceRecognition_ImagePreprocessing import face_recording
from FaceRecognition_plotting import plotReconstructionManual, plotTSNE, plotPCA2components, plotEigenfaces, \
    plotPCAcomponents, create_plots




class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        # Enable access to inherieted variables, methods, classes etc.
        super(self.__class__, self).__init__() # Inheritance

        ################## Initialize the application ##################
        self.setupUi(self)

        ################## Cosmetic settings ##################
        # Setting up icon and window title
        self.setWindowTitle('Basic Face Recognition System')
        self.setWindowIcon(QIcon('GUI_Components//app_icon.jpg'))

        # ############### Plot embedded display ################
        self.LearnEigenfaces.clicked.connect(self.DatabaseEigenfaces)
        self.AddPersonButton.clicked.connect(self.addPerson)
        self.IdentifyButton.clicked.connect(self.identifyPerson)
        self.show()

        # self.DatabaseEigenfaces()
        # confidence, person_id, im_searched, im_found, im_found_id = self.efr.recognize_face(gui=self)
        #
        # # Message box
        # self.msg = QMessageBox()
        # self.msg.setIcon(QMessageBox.Information)
        # self.msg.setWindowTitle("Twarz została odnaleziona")
        # self.msg.setText("Pewnosc jest, nie ma?")
        # self.msg.exec_()
        #
        # # Message box
        # self.msg = QMessageBox()
        # self.msg.setIcon(QMessageBox.Information)
        # self.msg.setWindowTitle("Twarz została odnaleziona")
        # self.msg.setText("Pewnosc jest, nie ma?")
        # self.msg.exec_()
        #
        # # Transfer numpy arrays to QImages
        # im_searched = np.array(im_searched).astype(np.int32)
        # qimage1 = QImage(im_searched, im_searched.shape[0], im_searched.shape[1],
        #                 QImage.Format_RGB32)
        # pixmap1 = QPixmap(qimage1)
        # pixmap1 = pixmap1.scaled(320, 240, Qt.KeepAspectRatio)
        # self.IdentifySearchLabel.setPixmap(pixmap1)
        #
        # qimage2 = QImage(im_found, im_found.shape[1], im_found.shape[0],
        #                 QImage.Format_RGB888)
        # pixmap2 = QPixmap(qimage2)
        # pixmap2 = pixmap2.scaled(320, 240, Qt.KeepAspectRatio)
        # self.IdentifyFoundLabel.setPixmap(pixmap2)


    def identifyPerson(self):
        confidence, person_id, im_searched, im_found, im_found_id = self.efr.recognize_face(gui=self)

        # Message box
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setWindowTitle("Twarz została odnaleziona")
        self.msg.setText("Pewnosc jest, nie ma?")
        self.msg.exec_()

        # Message box
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setWindowTitle("Twarz została odnaleziona")
        self.msg.setText("Pewnosc jest, nie ma?")
        self.msg.exec_()

        # Transfer numpy arrays to QImages
        qimage1 = QImage(im_searched, im_searched.shape[1], im_searched.shape[0],
                        QImage.Format_RGB888)
        pixmap1 = QPixmap(qimage1)
        # pixmap1 = pixmap1.scaled(320, 240, Qt.KeepAspectRatio)
        self.IdentifySearchLabel.setPixmap(pixmap1)

        qimage2 = QImage(im_searched, im_searched.shape[1], im_searched.shape[0],
                        QImage.Format_RGB888)
        pixmap2 = QPixmap(qimage2)
        # pixmap2 = pixmap2.scaled(320, 240, Qt.KeepAspectRatio)
        self.IdentifyFoundLabel.setPixmap(pixmap2)


    def addPerson(self):
        self.efr.add_person(gui=self)
        self.PlotEigenfacesData()

        # Message box
        self.msg = QMessageBox()
        self.msg.setIcon(QMessageBox.Information)
        self.msg.setWindowTitle("Dodawanie twarzy zakończone")
        self.msg.setText("Baza twarzy została zaktualizowana.\nWizualizacja danych zawiera teraz nowe dane.")
        self.msg.exec_()


    def DatabaseEigenfaces(self):
        self.fr = FaceRecognitionEigenfaces()
        self.fr.get_images()
        self.fr.get_eigenfaces()
        self.efr = EigenfaceRecognitionNewfaces(data=self.fr)

        self.PlotEigenfacesData()

    def PlotEigenfacesData(self):
        self.fr.stochastic_neighbour_embedding()
        plotPCAcomponents(self)
        plotEigenfaces(self)
        plotPCA2components(self)
        plotTSNE(self)
        plotReconstructionManual(self)



if __name__ == '__main__':        # if we're running file directly and not importing it
    app = QApplication(sys.argv)  # A new instance of QApplication
    form = MainWindow()  # New instance of application
    form.show()  # Show the form
    app.exec_()  # and execute the app