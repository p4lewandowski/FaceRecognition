import sys, math
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from EarRecognition_gui import Ui_MainWindow
from EarRecognition_eigenears_core import EarRecognitionEigenears
from EarRecognition_eigenears_recognition import EigenearsRecognition
from EarRecognition_cnn import Cnn_model
from EarRecognition_plotting import plotReconstructionManual, plotTSNE, plotPCA2components, plotEigenears, \
    plotPCAcomponents, show_found_ear, create_messagebox
from EarRecognition_cnn_stream import GUI_ConsoleOutput
from EarRecognition_imagepreprocessing import bulk_ear_detection, bulk_ear_visualization, bulk_identify_ears

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Szkielet aplikacji. Inicjalizuje GUI a także rozporządza poszczególnymi zaimportowanymi funkcjami.
    Klasa po wywołaniu inicjalizuje klasy dotyczące obliczeń związanych z sieciami neuronowymi i
    uszami własnymi. Dziedziczy GUI i przypisuje przyciskom różne funkcje.
    """

    def __init__(self):
        # Dziedzicz z QMainWindow i Ui_MainWindow
        super(self.__class__, self).__init__()

        ################## Zainicjalizuj aplikację ##################
        self.setupUi(self)

        ################## Zmiany kosmetyczne ##################
        # Ikony i tytuł okna
        self.setWindowTitle('System rozpoznawania osób na podstawie ucha')
        self.setWindowIcon(QIcon('app_icon.jpg'))

        ################## Zdjęcia do wizualizacji metod ##################
        pixmap_eigen = QPixmap('eigenears.jpg').scaledToWidth(325)
        self.Eigenfaces_label.setPixmap(pixmap_eigen)
        pixmap_neural = QPixmap('neural_networks_g.jpg').scaledToHeight(320)
        self.Neuralnetworks_label.setPixmap(pixmap_neural)

        ############### Połącz przyciski ################
        self.LearnEigenfaces.clicked.connect(self.DatabaseEigenears)
        self.AddPersonButton.clicked.connect(self.addPerson)
        self.IdentifyButton.clicked.connect(self.identifyPerson)
        self.WelcomeButton.clicked.connect(self.turnDatabaseTab)
        self.LearnNetworks.clicked.connect(self.TrainCNN)

        self.LoadImButton.clicked.connect(self.Bulk_LoadImage)
        self.IdentifyPeopleViewButton_original.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.IdentifyPeopleViewButton_ears.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.IdentifyPeopleViewButton_identified.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.IdentifyPeopleViewButton_covered.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
        self.ShowEarsButton.clicked.connect(self.Bulk_FindEar)
        self.IdentifyPeopleButton.clicked.connect(self.Bulk_IndetifyEars)

        ############### Stwórz klasy do obliczeń ###############
        self.er = EarRecognitionEigenears()
        self.cnn = Cnn_model()
        # Ustawianie informacji o ilości dodanych uszu
        self.cnn_label_state.setText("Trenuj model z użyciem {} dodanych uszu.".format(self.cnn.nb_class))
        myFont = QFont()
        myFont.setBold(True)
        self.cnn_label_state.setFont(myFont)

        self.show()

    def turnDatabaseTab(self):
        """
        Funkcja inicjalizowana przyciskiem, zmieniająca obecną zakładkę na zakładkę gdzie ładowana
        jest baza danych dla uszu własnych i gdzie trenowana jest sieć neuronowa.
        """
        self.tabWidget.setCurrentIndex(1)

    def identifyPerson(self):
        """
        Funkcja inicjalizowana przyciskiem, gdzie zależnie od opcji zaznaczonej na przycisku
        radiowym następuje identyfikacja osób z użyciem uszu własnych bądź z użyciem sieci
        neuronowej. Funkcja wyświetla również komunikaty zależne od stanu powodzenia funkcji.
        """
        if self.iden_radio_eigen.isChecked():
            try:
                confidence, person_id, im_searched, im_found, im_found_id = self.efr.recognize_ear(gui=self)
                if confidence:
                    # Jeśli znalezione ucho nie należy do nowo dodanych tylko do bazy poczatkowej
                    if im_found_id < self.efr.loaded_im_len:
                        create_messagebox(self, "Ucho zostało odnalezione",
                                          "Wyświetlona zostanie ucho z poczatkowej, wczytanej bazy danych.")
                    else:
                        # N - liczba zdjęć początkowych w bazie; K - ilość zdjęć wykonywanych dla nowej osoby
                        # E - indeks znalezionego zdjęcia
                        # ceil((E-N)/K) - 1 aby indeks zaczynał się od 0
                        id = math.ceil((im_found_id-self.efr.loaded_im_len)/self.efr.fotosperclass_number)-1
                        create_messagebox(self, "Twarz została odnaleziona",
                                      "Jest to ucho dodana kamerą. ID OSOBY: {}.".format(id))
                        show_found_ear(im_searched, im_found, id=id)
                        return

                else:
                    create_messagebox(self, "Ucho nie zostało odnalezione",
                                      "Dodane ucho nie zostało zaklasyfikowane.\n"
                                      "Nastąpi wyświetlenie najbardziej zbliżonego ucha.")

                show_found_ear(im_searched, im_found)

            except:
                create_messagebox(self, "Brak bazy danych", "Wczytaj bazę danych.")

        if self.iden_radio_nn.isChecked():
            if self.cnn.nb_class >= 2:
                im_searched, im_probabilities = self.cnn.recognize_ear(gui=self)
                create_messagebox(self, "Analiza zakończona",
                                  "Nastąpi wyświetlenie wyników.")
                show_found_ear(im_searched, [im_probabilities, self.cnn.img_data,
                    self.cnn.label_id], cnn=True)
            else:
                create_messagebox(self, "Brak modelu sieci.", "Dodaj osoby i wytrenuj model.")

    def addPerson(self):
        """
        Funkcja inicjalizowana przyciskiem, odpowiedzialna za dodanie nowej osoby(ucha). Zależnie od
        przycisku radiowego funkcja albo dodaje osobę do wcześniej wczytanej bazy danych związanej
        z uszami własnymi bądź dodaje osobę do zbioru danych związanego z sieciami neuronowymi do
        przetrenowania sieci w późniejszym czasie. Funkcja wyświetla komunikaty zależne od stanu
        powodzenia funkcji.
        """
        if self.add_radio_eigen.isChecked():
            try:
                self.efr.add_person(gui=self)
                self.PlotEigenearsData()
                # N - liczba zdjęć początkowych w bazie; K - ilość zdjęć wykonywanych dla nowej osoby
                # E - liczba wszystkich zdjęć
                # (E-N)/K - 1 aby indeks zaczynał się od 0
                create_messagebox(self, "Dodawanie ucha zakończone",
                                  "Baza uszu została zaktualizowana o nowe dane\nID OSOBY = {}.".format(
                                      int((len(self.efr.ear_data.labels) - self.efr.loaded_im_len) /
                                          self.efr.fotosperclass_number - 1)))
            except:
                create_messagebox(self, "Brak bazy danych", "Wczytaj bazę danych.")

        if self.add_radio_nn.isChecked():
            self.cnn.add_person(gui=self)
            create_messagebox(self, "Dodawanie ucha zakończone",
                              "Dane do sieci neuronowej zostały dodane.\nTrenuj model albo dodaj więcej osób.\n"
                              "ID OSOBY: {}".format(self.cnn.label_id[-1]))
            self.cnn_label_state.setText("Trenuj model z użyciem {} dodanych osób.".format(self.cnn.nb_class))

    def DatabaseEigenears(self):
        """
        Funkcja wywoływana przyciskiem. Wczytuje ona bazę danych dotyczącą uszu własnych a
        następnie wyznacza uszy własne i wywołuje funkcję do wizualizacji obliczonych danych.
        """
        try:
            self.er.get_images()
            self.er.get_eigenears()
            self.efr = EigenearsRecognition(data=self.er)

            self.PlotEigenearsData()
            create_messagebox(self, "Baza danych została wczytana", "Wizualizacja danych dostępna.")
        except:
            create_messagebox(self, "BŁĄD!",
                              "Brak zdjęć do obliczenia uszu własnych.")

    def TrainCNN(self):
        """
        Funkcja wywoływana przyciskiem. Gdy więcej niż jedno ucho z zostało dodane do przetrenowania
        dla sieci neuronowej funkcja inicjalizuje trening sieci a także tworzy nowe okno prezentujące
        proces treningu sieci. Wyświetlane są komunikaty zależne od stanu powodzenia funkcji.
        """
        if self.cnn.nb_class < 2:
            create_messagebox(self, "Niemożliwy trening sieci", "Baza zawiera za mało osób do treningu.")
        else:
            self.out_stream = GUI_ConsoleOutput()
            self.cnn.data_processing()
            self.cnn.model_compile()
            self.cnn.train_cnn()
            self.out_stream.btn.setEnabled(True)
            create_messagebox(self, "Sieć przetrenowana.", "Możliwa identyfikacja osób.")
            if self.ShowEarsButton.isEnabled():
                self.IdentifyPeopleButton.setEnabled(1)
            self.NetworkTrained = True

    def PlotEigenearsData(self):
        """
        Funkcja wywoływana automatycznie po wczytaniu bazy dotyczacej uszu własnych bądź w
        przypadku dodania nowej osoby do bazy danych.
        """
        self.er.stochastic_neighbour_embedding()
        plotPCA2components(self)
        plotPCAcomponents(self)
        plotEigenears(self)
        plotTSNE(self)
        plotReconstructionManual(self)

    def Bulk_LoadImage(self):
        """
        Funkcja wczytująca plik zdjęciowy a następnie wyświetlająca zdjęcie w aplikacji.
        """
        filter = "Images (*.png *.jpg)"
        image_obj, _ = QFileDialog.getOpenFileName(self, 'Open image', 'Desktop', filter)
        if image_obj:
            self.bulk_image = bulk_ear_visualization(image_obj, self)
            self.IdentifyPeopleViewButton_original.setEnabled(1)
            self.ShowEarsButton.setEnabled(1)
            self.IdentifyPeopleViewButton_ears.setEnabled(0)
            self.IdentifyPeopleViewButton_identified.setEnabled(0)
            self.IdentifyPeopleViewButton_covered.setEnabled(0)
            self.IdentifyPeopleButton.setEnabled(0)
            self.stackedWidget.setCurrentIndex(0)


    def Bulk_FindEar(self):
        """
        Funkcja zaznaczająca twarze na wcześniej wczytanym zdjęciu.
        """
        bulk_ear_detection(self)
        try:
            if self.NetworkTrained:
                self.IdentifyPeopleViewButton_ears.setEnabled(1)
                self.IdentifyPeopleButton.setEnabled(1)
        except:
            None
        self.stackedWidget.setCurrentIndex(1)
        self.IdentifyPeopleViewButton_ears.setEnabled(1)

    def Bulk_IndetifyEars(self):
        """
        Funkcja nadpisująca oryginalne twarze wykrytymi twarzami a także wyświetlająca wartości
        procentowe uzyskane przez sieć neuronową co do poprawności identyfikacji.
        Funkcja dostepna wyłącznie po przetrenowaniu sieci i dodaniu zdjęcia.
        """
        bulk_identify_ears(self, self.cnn.fotosperclass_number)
        self.IdentifyPeopleViewButton_identified.setEnabled(1)
        self.IdentifyPeopleViewButton_covered.setEnabled(1)
        self.stackedWidget.setCurrentIndex(2)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()