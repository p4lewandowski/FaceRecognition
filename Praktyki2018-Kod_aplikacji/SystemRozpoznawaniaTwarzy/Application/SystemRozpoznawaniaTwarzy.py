import sys, math
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from FaceRecognition_gui import Ui_MainWindow
from FaceRecognition_eigenfaces_core import FaceRecognitionEigenfaces
from FaceRecognition_eigenfaces_recognition import EigenfaceRecognition
from FaceRecognition_cnn import Cnn_model
from FaceRecognition_plotting import plotReconstructionManual, plotTSNE, plotPCA2components, plotEigenfaces, \
    plotPCAcomponents, show_found_face, create_messagebox
from FaceRecognition_cnn_stream import GUI_ConsoleOutput
from FaceRecognition_imagepreprocessing import bulk_face_detection, bulk_face_visualization, bulk_identify_faces

class MainWindow(QMainWindow, Ui_MainWindow):
    """
    Szkielet aplikacji. Inicjalizuje GUI a także rozporządza poszczególnymi zaimportowanymi funkcjami.
    Klasa po wywołaniu inicjalizuje klasy dotyczące obliczeń związanych z sieciami neuronowymi i
    twarzami własnymi. Dziedziczy GUI i przypisuje przyciskom różne funkcje.
    """

    def __init__(self):
        # Dziedzicz z QMainWindow i Ui_MainWindow
        super(self.__class__, self).__init__()

        ################## Zainicjalizuj aplikację ##################
        self.setupUi(self)

        ################## Zmiany kosmetyczne ##################
        # Ikony i tytuł okna
        self.setWindowTitle('System rozpoznawania twarzy')
        self.setWindowIcon(QIcon('app_icon.jpg'))

        # Zdjęcia do wizualizacji metod
        pixmap_eigen = QPixmap('eigenfaces.jpg').scaledToWidth(325)
        self.Eigenfaces_label.setPixmap(pixmap_eigen)
        pixmap_neural = QPixmap('neural_networks_g.jpg').scaledToHeight(320)
        self.Neuralnetworks_label.setPixmap(pixmap_neural)
        # Ustaw domyślne wyświetlanie w Identyfikacje ze zdjęcia
        self.stackedWidget.setCurrentIndex(0)

        ############### Połącz przyciski ################
        self.LearnEigenfaces.clicked.connect(self.DatabaseEigenfaces)
        self.AddPersonButton.clicked.connect(self.addPerson)
        self.IdentifyButton.clicked.connect(self.identifyPerson)
        self.WelcomeButton.clicked.connect(self.turnDatabaseTab)
        self.LearnNetworks.clicked.connect(self.TrainCNN)

        self.LoadImButton.clicked.connect(self.Bulk_LoadImage)
        self.IdentifyPeopleViewButton_original.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.IdentifyPeopleViewButton_faces.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.IdentifyPeopleViewButton_identified.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.IdentifyPeopleViewButton_covered.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(3))
        self.ShowFacesButton.clicked.connect(self.Bulk_FindFace)
        self.IdentifyPeopleButton.clicked.connect(self.Bulk_IndetifyFaces)

        ############### Stwórz klasy do obliczeń ###############
        self.fr = FaceRecognitionEigenfaces()
        self.cnn = Cnn_model()
        # Ustawienie tekstu o ilości dodanych osób
        self.cnn_label_state.setText("Trenuj model z użyciem {} dodanych osób.".format(self.cnn.nb_class))
        myFont = QFont()
        myFont.setBold(True)
        self.cnn_label_state.setFont(myFont)

        self.show()

    def turnDatabaseTab(self):
        """
        Funkcja inicjalizowana przyciskiem, zmieniająca obecną zakładkę na zakładkę gdzie ładowana
        jest baza danych dla twarzy własnych i gdzie trenowana jest sieć neuronowa.
        """
        self.tabWidget.setCurrentIndex(1)

    def identifyPerson(self):
        """
        Funkcja inicjalizowana przyciskiem, gdzie zależnie od opcji zaznaczonej na przycisku
        radiowym następuje identyfikacja twarzy z użyciem twarzy własnych bądź z użyciem sieci
        neuronowej. Funkcja wyświetla również komunikaty zależne od stanu powodzenia funkcji.
        ID osoby jest pokazywane w oknie komunikatu bądź bezpośrednio na zdjęciu zależnie
        od użytej metody.
        """
        if self.iden_radio_eigen.isChecked():
            try:
                confidence, face_id, im_searched, im_found, im_found_id = self.efr.recognize_face(gui=self)
                if confidence:
                    # Jeśli znaleziona twarz nie należy do nowo dodanych tylko do bazy poczatkowej
                    if im_found_id < self.efr.loaded_im_len:
                        create_messagebox(self, "Twarz została odnaleziona",
                                          "Wyświetlona zostanie osoba w bazie danych.\n"
                                          "Jest to osoba z wczytanej bazy danych.")
                    else:
                        # N - liczba zdjęć początkowych w bazie; K - ilość zdjęć wykonywanych dla nowej osoby
                        # E - indeks znalezionego zdjęcia
                        # ceil((E-N)/K) - 1 aby indeks zaczynał się od 0
                        id = math.ceil((im_found_id-self.efr.loaded_im_len)/self.efr.fotosperclass_number)-1
                        create_messagebox(self, "Twarz została odnaleziona",
                                      "Jest to osoba dodana kamerą. ID OSOBY: {}.".format(id))
                        show_found_face(im_searched, im_found, id=id)
                        return

                else:
                    create_messagebox(self, "Twarz nie została odnaleziona",
                                      "Dodana twarz nie została zaklasyfikowana.\n"
                                      "Nastąpi wyświetlenie najbardziej zbliżonej osoby.")

                show_found_face(im_searched, im_found)
            except:
                create_messagebox(self, "Brak bazy danych", "Wczytaj bazę danych.")

        if self.iden_radio_nn.isChecked():
            if self.cnn.nb_class >= 2:
                im_searched, im_probabilities = self.cnn.recognize_face(gui=self)
                create_messagebox(self, "Analiza zakończona",
                                  "Nastąpi wyświetlenie wyników.")
                show_found_face(im_searched, [im_probabilities, self.cnn.img_data,
                    self.cnn.label_id], cnn=True)
            else:
                create_messagebox(self, "Brak modelu sieci.", "Dodaj osoby i wytrenuj model.")

    def addPerson(self):
        """
        Funkcja inicjalizowana przyciskiem, odpowiedzialna za dodanie nowej osoby. Zależnie od
        przycisku radiowego funkcja albo dodaje osobę do wcześniej wczytanej bazy danych związanej
        z twarzami własnymi bądź dodaje osobę do zbioru danych związanego z sieciami neuronowymi do
        przetrenowania sieci w późniejszym czasie. Funkcja wyświetla komunikaty zależne od stanu
        powodzenia funkcji.
        Po dodaniu osoby wyświetlone zostanie ID nowo dodanej osoby.
        """
        if self.add_radio_eigen.isChecked():
            try:
                self.efr.add_person(gui=self)
                self.PlotEigenfacesData()
                # N - liczba zdjęć początkowych w bazie; K - ilość zdjęć wykonywanych dla nowej osoby
                # E - liczba wszystkich zdjęć
                # (E-N)/K - 1 aby indeks zaczynał się od 0
                create_messagebox(self, "Dodawanie twarzy zakończone",
                                  "Baza twarzy została zaktualizowana o nowe dane\nID OSOBY = {}.".format(
                                      int((len(self.efr.face_data.labels)-self.efr.loaded_im_len)/
                                          self.efr.fotosperclass_number-1)))

            except:
                create_messagebox(self, "Brak bazy danych", "Wczytaj bazę danych.")

        if self.add_radio_nn.isChecked():
            self.cnn.add_person(gui=self)
            create_messagebox(self, "Dodawanie twarzy zakończone",
                              "Dane do sieci neuronowej zostały dodane.\nTrenuj model albo dodaj więcej osób.\n"
                              "ID OSOBY: {}".format(self.cnn.label_id[-1]))
            self.cnn_label_state.setText("Trenuj model z użyciem {} dodanych osób.".format(self.cnn.nb_class))

    def DatabaseEigenfaces(self):
        """
        Funkcja wywoływana przyciskiem. Wczytuje ona bazę danych dotyczącą twarzy własnych a
        następnie wyznacza twarze i wywołuje funkcję do wizualizacji obliczonych danych.
        """
        try:
            self.fr.get_images()
            self.fr.get_eigenfaces()
            self.efr = EigenfaceRecognition(data=self.fr)

            self.PlotEigenfacesData()
            create_messagebox(self, "Baza danych została wczytana", "Wizualizacja danych dostępna.")
        except:
            create_messagebox(self, "BŁĄD!",
                              "Brak zdjęć do obliczenia twarzy własnych.")

    def TrainCNN(self):
        """
        Funkcja wywoływana przyciskiem. Gdy więcej niż jedna twarz została dodana do przetrenowania
        dla sieci neuronowej funkcja inicjalizuje trening sieci a także tworzy nowe okno prezentujące
        proces treningu sieci. Funkcja wyświetla komunikaty zależne od stanu powodzenia funkcji.
        """
        if self.cnn.nb_class < 2:
            create_messagebox(self, "Niemożliwy trening sieci", "Baza zawiera za mało osób do treningu.")
        else:
            self.out_stream = GUI_ConsoleOutput()
            self.cnn.data_processing()
            self.cnn.initialize_networkmodel()
            self.cnn.train_cnn()
            self.out_stream.btn.setEnabled(True)
            create_messagebox(self, "Sieć przetrenowana.", "Możliwa identyfikacja osób.")
            # Odblokuj identyfikację w zdjęciach grupowych
            if self.ShowFacesButton.isEnabled():
                self.IdentifyPeopleButton.setEnabled(1)
            self.NetworkTrained = True

    def PlotEigenfacesData(self):
        """
        Funkcja wywoływana automatycznie po wczytaniu bazy dotyczacej twarzy własnych bądź w
        przypadku dodania nowej osoby do bazy danych.
        """
        self.fr.stochastic_neighbour_embedding()
        plotPCAcomponents(self)
        plotEigenfaces(self)
        plotPCA2components(self)
        plotTSNE(self)
        plotReconstructionManual(self)

    def Bulk_LoadImage(self):
        """
        Funkcja wczytująca plik zdjęciowy a następnie wyświetlająca zdjęcie w aplikacji.
        """
        filter = "Images (*.png *.jpg)"
        image_obj, _ = QFileDialog.getOpenFileName(self, 'Open image', 'Desktop', filter)
        if image_obj:
            self.bulk_image = bulk_face_visualization(image_obj, self)
            self.IdentifyPeopleViewButton_original.setEnabled(1)
            self.ShowFacesButton.setEnabled(1)
            self.IdentifyPeopleViewButton_faces.setEnabled(0)
            self.IdentifyPeopleViewButton_identified.setEnabled(0)
            self.IdentifyPeopleViewButton_covered.setEnabled(0)
            self.IdentifyPeopleButton.setEnabled(0)
            self.stackedWidget.setCurrentIndex(0)


    def Bulk_FindFace(self):
        """
        Funkcja zaznaczająca twarze na wcześniej wczytanym zdjęciu.
        """
        bulk_face_detection(self)
        try:
            if self.NetworkTrained:
                self.IdentifyPeopleViewButton_faces.setEnabled(1)
                self.IdentifyPeopleButton.setEnabled(1)
        except:
            None
        self.stackedWidget.setCurrentIndex(1)
        self.IdentifyPeopleViewButton_faces.setEnabled(1)

    def Bulk_IndetifyFaces(self):
        """
        Funkcja nadpisująca oryginalne twarze wykrytymi twarzami a także wyświetlająca wartości
        procentowe uzyskane przez sieć neuronową co do poprawności identyfikacji.
        Funkcja dostepna wyłącznie po przetrenowaniu sieci i dodaniu zdjęcia.
        """
        bulk_identify_faces(self, self.cnn.fotosperclass_number)
        self.IdentifyPeopleViewButton_identified.setEnabled(1)
        self.IdentifyPeopleViewButton_covered.setEnabled(1)
        self.stackedWidget.setCurrentIndex(2)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    form = MainWindow()
    form.show()
    app.exec_()