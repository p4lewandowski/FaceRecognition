import sys
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow

from GUI_Components.gui import Ui_MainWindow
from FaceRecognition_eigenfaces_core import FaceRecognitionEigenfaces
from FaceRecognition_eigenfaces_recognition import EigenfaceRecognition
from FaceRecognition_plotting import plotReconstructionManual, plotTSNE, plotPCA2components, plotEigenfaces, \
    plotPCAcomponents, create_plots, show_found_face, create_messagebox


class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        # Enable access to inherieted variables, methods, classes etc.
        super(self.__class__, self).__init__() # Inheritance

        ################## Initialize the application ##################
        self.setupUi(self)

        ################## Cosmetic settings ##################
        # Setting up icon and window title
        self.setWindowTitle('System rozpoznawania twarzy')
        self.setWindowIcon(QIcon('GUI_Components//app_icon.jpg'))

        # Setting up images for database choice
        pixmap_eigen = QPixmap('GUI_Components\\eigenfaces.png').scaledToWidth(250)
        self.Eigenfaces_label.setPixmap(pixmap_eigen)
        pixmap_neural = QPixmap('GUI_Components\\neural_networks_g.jpg').scaledToWidth(250)
        self.Neuralnetworks_label.setPixmap(pixmap_neural)

        # ############### Plot embedded display ################
        self.LearnEigenfaces.clicked.connect(self.DatabaseEigenfaces)
        self.AddPersonButton.clicked.connect(self.addPerson)
        self.IdentifyButton.clicked.connect(self.identifyPerson)
        self.WelcomeButton.clicked.connect(self.turnDatabaseTab)
        self.show()


    def turnDatabaseTab(self):
        self.tabWidget.setCurrentIndex(1)

    def identifyPerson(self):
        try:
            if self.iden_radio_eigen.isChecked():
                confidence, person_id, im_searched, im_found, im_found_id = self.efr.recognize_face(gui=self)
                if confidence:
                    create_messagebox(self, "Twarz została odnaleziona",
                                      "Wyświetlona zostanie najbardziej podobna twarz.")
                else:
                    create_messagebox(self, "Twarz nie została odnaleziona",
                                      "Dodana twarz nie została zaklasyfikowana.\n"
                                      "Nastąpi wyświetlenie najbardziej zbliżonej twarzy.")

                show_found_face(im_searched, im_found)
            if self.iden_radio_nn.isChecked():
                pass
        except:
            create_messagebox(self, "Brak bazy danych", "Wczytaj bazę danych.")


    def addPerson(self):
        try:
            if self.add_radio_eigen.isChecked():
                self.efr.add_person(gui=self)
                self.PlotEigenfacesData()
            if self.add_radio_nn.isChecked():
                pass

            create_messagebox(self, "Dodawanie twarzy zakończone",
                          "Baza twarzy została zaktualizowana.\nWizualizacja danych zawiera teraz nowe dane.")
        except:
            create_messagebox(self, "Brak bazy danych", "Wczytaj bazę danych.")

    def DatabaseEigenfaces(self):
        self.fr = FaceRecognitionEigenfaces()
        self.fr.get_images()
        self.fr.get_eigenfaces()
        self.efr = EigenfaceRecognition(data=self.fr)

        self.PlotEigenfacesData()
        create_messagebox(self, "Baza danych została wczytana", "Wizualizacja danych dostępna.")

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