import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QDialog
from gui import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from FaceRecognition_eigenfaces import FaceRecognitionEigenfaces
from FaceRecognition_newfaces import EigenfaceRecognitionNewfaces
from FaceRecognition_ImagePreprocessing import image_selection
import seaborn as sns
import random

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self):
        # Enable access to inherieted variables, methods, classes etc.
        super(self.__class__, self).__init__() # Inheritance

        ################## Initialize the application ##################
        self.setupUi(self)

        # ############### Plot embedded display ################
        # self.figure = plt.figure()
        # self.canvas = FigureCanvas(self.figure)
        # self.toolbar = NavigationToolbar(self.canvas, self)
        # self.toolbar.setParent(self.VisualizationTab)


        self.LearnEigenfaces.clicked.connect(self.plot)

    def plot(self):
        data = [random.random() for i in range(10)]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(data, '*-')
        self.canvas.draw()
        print(data)


        fr = FaceRecognitionEigenfaces()
        fr.get_images()
        fr.get_eigenfaces()

        # X = list(range(1, fr.eigenfaces_n + 1))
        # plot = sns.barplot(x=X, y=fr.explained_variance_ratio_, ax=self.ax)
        # plot.set_xlabel('N-ty główny komponent')
        # plot.set_ylabel('Wartosc procentowa reprezentowanej wariancji')
        # plot.set_title('Procentowa reprezentacja wariancji dla poszczególnych głównych składowych')



if __name__ == '__main__':        # if we're running file directly and not importing it
    app = QApplication(sys.argv)  # A new instance of QApplication
    form = MainWindow()  # New instance of application
    form.show()  # Show the form
    app.exec_()  # and execute the app