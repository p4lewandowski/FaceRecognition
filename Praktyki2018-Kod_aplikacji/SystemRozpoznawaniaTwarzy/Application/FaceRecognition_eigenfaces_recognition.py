from FaceRecognition_imagepreprocessing import face_recording, take_image
from FaceRecognition_eigenfaces_core import FaceRecognitionEigenfaces

import os
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

class EigenfaceRecognition:
    """
    Klasa zawierająca funkcje dotyczące dodawania nowych osób do bazy danych a także ich
    identyfikacji. Używa danych z klasy *FaceRecognitionEigenfaces*. Po zainstancjowaniu
    inicjalizująca parametry do metody k-najbliższych sąsiadów. Wczytuje ona dane do klasy z
    wcześniej stworzonego obiektu *FaceRecognitionEigenfaces* z wyznaczonymi twarzami własnymi bądź
    z pliku zawierającego wcześniej zapisaną kopię takiego obiektu. Dane z klasy
    *FaceRecognitionEigenfaces* są przypisane do *face_data*. Liczba najbliższych sąsiadów dla
    metody k-nn wynosi 5.

    Args:
        filepath: Ścieżka bezwzględna do pliku.

        data: Obiekt klasy *FaceRecognitionEigenfaces* z wyznaczonymi twarzami własnymi.
    """

    def __init__(self, filepath=None, data=None):

        if filepath:
            self.face_data = pickle.load(open(filepath, "rb"))
        if data:
            self.face_data = data
        # Zainicjalizuj parametry dla obiektu knn
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier.fit(self.face_data.face_weights, self.face_data.labels)
        self.newfacedir = os.path.join(self.face_data.datadir, 'new_faces_notindetected')
        self.loaded_im_len = len(self.face_data.labels)
        self.fotosperclass_number = 20



    def recognize_face(self, **kwargs):
        """
        Funkcja umożliwiająca rozpoznawanie twarzy z użyciem twarzy własnych oraz metody
        k-najbliższych sąsiadów. Jest ona również odpowiedzialna za wywołanie funkcji, która zrobi
        zdjęcie nowej osobie a także je przetworzy. Funkcja determinuje czy osoba dana na zdjęciu
        jest w zbiorze (tj. czy została rozpoznana) na podstawie dwóch czynników:
            - należy sprawdzić średnią odległość euklidesową zdjęć osoby A (wybranej przez knn) do
              nowo dodanego zdjęcia, które były brane pod uwagę przez k-nn. Jeśli odległość
              euklidesowa do nowo dodanego zdjęcia od innej osoby będzie mniejsza niż odległość
              klasy A pomnożona przez 1.6 to nowo dodana twarz jest traktowana jako nowa twarz nie
              będąca w bazie.
                Przykład:
                Mając 3 zdjęcia osoby A i 2 zdjęcia osoby B najbliższe nowo dodanemu zdjęciu jeśli
                średnia odległość zdjęć osoby A wynosi 100, a osoby B 150, to nowo dodane zdjęcie
                będzie zaklasyfikowane jako nie będące w bazie (nowa osoba) jako że 100 * 1.6 nie
                jest mniejsze niż 150.

            - jeśli dla 5 najbliższych sąsiadów będzie więcej niż 3 róznych kandydatów to nowo dodane
              zdjęcie klasyfikowane jest jako nowa osoba.

        Args:
            **kwargs:
                gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych
                z kamery w aplikacji GUI.

        Returns:
            boolean: True - Jeśli twarz została uznana za będącą w bazie danych. False - jeśli nie.

            face_found_id: indeks odnoszący się do osoby uznanej za najbardziej podobną do nowo
            dodanego zdjęcia. Odnosi się on do tablicy *label_id* z obiektu *FaceRecognitionEigenfaces*. ID klasy.

            image: zdjęcie nowo dodanej osoby.

            closest_face: zdjęcie, które zostało wytypowane jako "najbliższe" do nowo dodanej osoby.

            closest_face_id: indeks "najbliższego zdjęcia. Odnosi się on do tablicy *image_matrix_raw*
            z obiektu *FaceRecognitionEigenfaces*.
        """

        # Umożliw wyświetlanie w GUI jśli przekazano parametr
        gui = kwargs.get('gui', False)
        # Zrób zdjęcie, znajdź twarz i oblicz jego reprezentację
        image = take_image(gui=gui)
        image_representation = self.face_data.transfer_image(image.flatten())

        # Uaktualnij dane do knn (po nowo dodanej twarzy)
        self.knn_classifier.fit(self.face_data.face_weights, self.face_data.labels)
        # Policz prawdopodobieństwo dla knn
        probabilities = self.knn_classifier.predict_proba(image_representation.reshape(1, -1))
        prob_person = np.argsort(probabilities)[0][-3:]
        prob_val = probabilities[0][prob_person]

        isnew = 0
        dist, ids = self.knn_classifier.kneighbors(X=image_representation.reshape(1, -1),
                                                   n_neighbors=5, return_distance=True)
        person_ids = self.face_data.labels[ids]
        candidates_n = len([1 for x in prob_val if x > 0])

        class_distances = sum_class_distances(dist, person_ids)
        class_distances = sorted(class_distances, key=lambda x: x[1])

        # Jeśli jeden kandydat
        if candidates_n == 1:
            face_found_id = prob_person[-1]
        # Jeśli więcej
        else:
            # Jesli za dużo - nie można stwierdzić czyja to twarz
            if candidates_n >= 4:
                isnew = 1

            # Sprawdź czy można stwierdzić z pewnością czyja to twarz
            # Jeśli więcej niż jeden kandydat
            elif candidates_n != 1:
                # Jeśli odległość pierwszej klasy *1.6 jest większa niż odległość drugiej klasy
                # - twrz nie jest zaklasyfikowana z "pewnością"
                if class_distances[1][1][0] < class_distances[0][1][0] * 1.6:
                    isnew = 1

            # Wybierz klasę bardziej prawdopodobną (lista posortowana od najmniejszego prawdop.)
            if prob_val[-1] > prob_val[-2]:
                face_found_id = prob_person[-1]
                # Jeśli klasa została wybrana, a twarz innej klasy jako zdjęcie i tak było bliżej
                # - mimo, że większość decyduje - jest to "niepewny" wybór
                if prob_person[-1] != person_ids[0][0]:
                    isnew=1
            # Jeśli prawdopodobieństwo takie samo - wybierz bliższą w kontekście dystansu klasy do twarzy
            elif prob_val[-1] == prob_val[-2]:
                if class_distances[-1][1][0] < class_distances[-2][1][0]:
                    face_found_id = prob_person[-1]
                else:
                    face_found_id = prob_person[-2]

        # Wybranie twarzy do wyświetlenia jako znaleziona
        if person_ids[0][0] == face_found_id:
            closest_face = np.reshape(self.face_data.image_matrix_raw.T[ids[0][0]],
                                      (self.face_data.image_shape, self.face_data.image_shape,))
            closest_face_id = ids[0][0]
        else:
            closest_face = np.reshape(self.face_data.image_matrix_raw.T[ids[0][1]],
                                      (self.face_data.image_shape, self.face_data.image_shape,))
            closest_face_id = ids[0][1]


        if isnew:
            # print("Face found without confidence, label = {}".format(face_found_id))
            return False, face_found_id, image, closest_face, closest_face_id

        if not isnew:
            # print("Face found with confidence, label = {}".format(face_found_id))
            return True, face_found_id, image, closest_face, closest_face_id


    def add_person(self, **kwargs):
        """
        Funkcja umożliwiająca dodanie nowej osoby do bazy danych, z której zostaną wyznaczone twarze
        własne. Jest ona odpowiedzialna za wywołanie funkcji, która robi i przetwarza zdjęcia nowej
        osoby. Przyjmuje ona opcjonalny argument gui, który umożliwia wyświetlanie danych z kamery w
        aplikacj GUI.

        Args:
            **kwargs:
                gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych
                z kamery w aplikacji GUI.
        """
        # Jeśli przekazano - użyj GUI
        gui = kwargs.get('gui', False)
        data = face_recording(gui=gui, im_count=self.fotosperclass_number)
        data = np.array(data)

        # Znajdź nową etykietę i zaktualizuj bazę
        label = max(self.face_data.labels) + 1
        count = len(data)
        data = data.reshape(count, self.face_data.image_shape**2)

        for image in data:
            self.add_image2database(image, label)


    def add_image2database(self, image, label):
        """
        Dodaje zdjęcia do bazy danych dotyczącej twarzy własnych z określoną etykietą.

        Args:
            image: "Spłaszczone" zdjęcie, jako wektor danych, które ma zostać dodane do bazy.

            label: Etykieta osoby, do której należy zdjęcie.
        """

        # Zwiększ licznik liczby zdjęć dla spójności
        self.face_data.image_count += 1
        # Dodaj etykietę
        self.face_data.labels = np.append(self.face_data.labels, label)

        # Zaktualizuj twarz średnią
        self.face_data.image_matrix_raw = np.vstack((self.face_data.image_matrix_raw.T, image)).T
        self.mean_img = np.sum(self.face_data.image_matrix_raw, axis=1) / self.face_data.image_count
        self.mean_img = self.mean_img.reshape(self.face_data.image_shape, self.face_data.image_shape)

        # Odejmij twarz średnią od nieprzetworzonych zdjęć wejściowych dla spójności
        self.face_data.image_matrix_flat = np.array(
            [x - self.mean_img.flatten() for x in self.face_data.image_matrix_raw.transpose()]).transpose()

        # Dodaj reprezentację twarzy do bazy
        self.face_data.face_weights = np.matmul(self.face_data.image_matrix_flat.transpose(),
                                                self.face_data.eigenfaces_flat.transpose())


def sum_class_distances(distances, class_labels):
    """
    Funkcja oblicza średni dystans od klas(osób) do nowo dodanej twarzy.

    Args:
        distances: Dystanse poszczególnych zdjęć twarzy w bazie do nowo dodanego zdjęcia.

        class_labels: Etykiety opisujące do której klasy należy zdjęcie w *distances*.

    Returns:
        arr: Lista elementów zawierających w każdym wierszu:
            - klasę do której należą zdjęcia,
            - średnią odległość twarz z danej klasy do nowo dodanej twarzy.
    """
    un_val = np.unique(class_labels)
    arr = []
    for i in un_val:
        sum = 0
        count = 0
        arr_pos = 0
        for elem in class_labels[0]:
            if elem==i:
                sum +=distances[0][arr_pos]
                count+=1
            arr_pos+=1
        arr.append(np.vstack((i, sum/count)))

    return arr

