import sys

from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
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
        self.show()


    def addPerson(self):
        efr = EigenfaceRecognitionNewfaces(data=self.fr)
        efr.add_person(gui=self)

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