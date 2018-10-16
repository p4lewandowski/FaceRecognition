from EarRecognition_imagepreprocessing import ear_recording, take_image
from EarRecognition_eigenears_core import EarRecognitionEigenears

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pickle

class EigenearsRecognition:
    """
    Klasa zawierająca funkcje dotyczące dodawania nowych osób do bazy danych a także ich
    identyfikacji. Używa danych z klasy *EarRecognitionEigenears*. Po zainstancjowaniu
    inicjalizująca parametry do metody k-najbliższych sąsiadów. Wczytuje ona dane do klasy z
    wcześniej stworzonego obiektu *EarRecognitionEigenears* z wyznaczonymi uszami własnymi bądź
    z pliku zawierającego wcześniej zapisaną kopię takiego obiektu. Dane z klasy
    *EarRecognitionEigenears* są przypisane do *ear_data*. Liczba najbliższych sąsiadów dla metody
     k-nn wynosi 5.

    Args:
        filepath: Ścieżka bezwzględna do pliku.

        data: Obiekt klasy *EarRecognitionEigenears* z wyznaczonymi uszami własnymi.
    """

    def __init__(self, filepath=None, data=None):

        if filepath:
            self.ear_data = pickle.load(open(filepath, "rb"))
        if data:
            self.ear_data = data
        # Zainicjalizuj parametry dla obiektu knn
        self.knn_classifier = KNeighborsClassifier(n_neighbors=5)
        self.knn_classifier.fit(self.ear_data.ear_weights, self.ear_data.labels)
        self.fotosperclass_number = 20
        self.loaded_im_len = len(self.ear_data.labels)


    def recognize_ear(self, **kwargs):
        """
        Funkcja umożliwiająca rozpoznawanie ucha z użyciem uszu własnych oraz metody
        k-najbliższych sąsiadów. Jest ona również odpowiedzialna za wywołanie funkcji, która zrobi
        zdjęcie nowej osobie a także je przetworzy. Funkcja determinuje czy osoba dana na zdjęciu
        jest w zbiorze (tj. czy została rozpoznana) na podstawie trzec czynników:
            - należy sprawdzić średnią odległość euklidesową zdjęć osoby A (wybranej przez knn)do
              nowo dodanego zdjęcia, które były brane pod uwagę przez k-nn. Jeśli odległość
              euklidesowa do nowo dodanego zdjęcia od innej osoby będzie mniejsza niż odległość
              klasy A pomnożona przez 1.6 to nowo dodane ucho jest traktowana jako nowe ucho nie
              będąca w bazie.
                Przykład:

                Mając 3 zdjęcia osoby A i 2 zdjęcia osoby B najbliższe nowo dodanemu zdjęciu jeśli
                średnia odległość zdjęć osoby A wynosi 100, a osoby B 150, to nowo dodane zdjęcie
                będzie zaklasyfikowane jako nie będące w bazie (nowa osoba) jako że 100 * 1.6 nie
                jest mniejsze niż 150.
            - jeśli dla 5 najbliższych sąsiadów będzie więcej niż 3 róznych kandydatów to nowo dodane
              zdjęcie klasyfikowane jest jako nowa osoba.
            - jeśli odległość od najbliższej klasy jest większa niż 2500 - niepewna identyfikacja
              bądź brak zdjęcia w zbiorze.

        Args:
            **kwargs:
                gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych
                z kamery w aplikacji GUI.

        Returns:
            boolean: True - Jeśli ucho zostało uznane za będącą w bazie danych. False - jeśli nie.

            ear_found_id: indeks odnoszący się do osoby uznanej za najbardziej podobną do nowo
            dodanego zdjęcia. Odnosi się on do tablicy *label_id* z obiektu *EarRecognitionEigenears*.

            image: zdjęcie nowo dodanej osoby.

            closest_ear: zdjęcie, które zostało wytypowane jako "najbliższe" do nowo dodanej osoby.

            closest_ear_id: indeks "najbliższego zdjęcia". Odnosi się on do tablicy *image_matrix_raw*
            z obiektu *EarRecognitionEigenears*.
        """

        # Umożliw wyświetlanie w GUI jśli przekazano parametr
        gui = kwargs.get('gui', False)
        # Zrób zdjęcie, znajdź ucho i oblicz jego reprezentację
        image = take_image(gui=gui)
        image_representation = self.ear_data.transfer_image(image.flatten())

        # Uaktualnij dane do knn (po nowo dodanym uchu)
        self.knn_classifier.fit(self.ear_data.ear_weights, self.ear_data.labels)
        # Policz prawdopodobieństwo dla knn
        probabilities = self.knn_classifier.predict_proba(image_representation.reshape(1, -1))
        prob_person = np.argsort(probabilities)[0][-3:]
        prob_val = probabilities[0][prob_person]

        isnew = 0
        dist, ids = self.knn_classifier.kneighbors(X=image_representation.reshape(1, -1),
                                                   n_neighbors=5, return_distance=True)
        person_ids = self.ear_data.labels[ids]
        candidates_n = len([1 for x in prob_val if x > 0])

        class_distances = sum_class_distances(dist, person_ids)
        class_distances = sorted(class_distances, key=lambda x: x[1])

        # Jeśli jeden kandydat
        if candidates_n == 1:
            ear_found_id = prob_person[-1]
        # Jeśli więcej
        else:
            # Jesli za dużo - nie można stwierdzić z pewnością przynależności ucha
            if candidates_n >= 3:
                isnew = 1

            # Sprawdź czy można stwierdzić z pewnością czyje to ucho
            # Jeśli więcej niż jeden kandydat
            elif candidates_n != 1:
                # Jeśli odległość pierwszej klasy *1.6 jest większa niż odległość drugiej klasy
                # - twrz nie jest zaklasyfikowana z "pewnością"
                if class_distances[1][1][0] < class_distances[0][1][0] * 1.6:
                    isnew = 1

            # Wybierz klasę bardziej prawdopodobną (lista posortowana od najmniejszego prawdop.)
            if prob_val[-1] > prob_val[-2]:
                ear_found_id = prob_person[-1]
                # Jeśli klasa została wybrana, a ucho innej klasy jako zdjęcie i tak było bliżej
                # - mimo, że większość decyduje - jest to "niepewny" wybór
                if prob_person[-1] != person_ids[0][0]:
                    isnew=1
            # Jeśli prawdopodobieństwo takie samo - wybierz bliższą w kontekście dystansu klasy do ucha
            elif prob_val[-1] == prob_val[-2]:
                if class_distances[-1][1][0] < class_distances[-2][1][0]:
                    ear_found_id = prob_person[-1]
                else:
                    ear_found_id = prob_person[-2]

        # Wybranie ucha do wyświetlenia jako znaleziona
        if person_ids[0][0] == ear_found_id:
            closest_ear = np.reshape(self.ear_data.image_matrix_raw.T[ids[0][0]],
                                      (self.ear_data.image_shape_one, self.ear_data.image_shape_two,))
            closest_ear_id = ids[0][0]
        else:
            closest_ear = np.reshape(self.ear_data.image_matrix_raw.T[ids[0][1]],
                                      (self.ear_data.image_shape_one, self.ear_data.image_shape_two,))
            closest_ear_id = ids[0][1]

        # Jeśli dystans większy niż 2500 - niepewna identyfikacja
        if class_distances[0][1]>2500:
            isnew=1


        if isnew:
            return False, ear_found_id, image, closest_ear, closest_ear_id

        if not isnew:
            return True, ear_found_id, image, closest_ear, closest_ear_id


    def add_person(self, **kwargs):
        """
        Funkcja umożliwiająca dodanie nowej osoby do bazy danych, z której zostaną wyznaczone uszy
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
        data = ear_recording(gui=gui)
        data = np.array(data)

        # Znajdź nową etykietę i zaktualizuj bazę
        label = max(self.ear_data.labels) + 1
        count = len(data)
        data = data.reshape(count, self.ear_data.image_shape_two * self.ear_data.image_shape_one)

        for image in data:
            self.add_image2database(image, label)


    def add_image2database(self, image, label):
        """
        Dodaje zdjęcia do bazy danych dotyczącej uszu własnych z określoną etykietą.

        Args:
            image: "Spłaszczone" zdjęcie, jako wektor danych, które ma zostać dodane do bazy.

            label: Etykieta osoby, do której należy zdjęcie.
        """

        # Zwiększ licznik liczby zdjęć dla spójności
        self.ear_data.image_count += 1
        # Dodaj etykietę
        self.ear_data.labels = np.append(self.ear_data.labels, label)

        # Zaktualizuj ucho średnie
        self.ear_data.image_matrix_raw = np.vstack((self.ear_data.image_matrix_raw.T, image)).T
        self.mean_img = np.sum(self.ear_data.image_matrix_raw, axis=1) / self.ear_data.image_count
        self.mean_img = self.mean_img.reshape(
            self.ear_data.image_shape_one, self.ear_data.image_shape_two)

        # Odejmij ucho średnie od nieprzetworzonych zdjęć wejściowych dla spójności
        self.ear_data.image_matrix_flat = np.array(
            [x - self.mean_img.flatten() for x in self.ear_data.image_matrix_raw.transpose()]).transpose()

        # Dodaj reprezentację ucha do bazy
        self.ear_data.ear_weights = np.matmul(self.ear_data.image_matrix_flat.transpose(),
                                               self.ear_data.eigenears_flat.transpose())


def sum_class_distances(distances, class_labels):
    """
    Funkcja oblicza średni dystans od klas(osób) do nowo dodanego ucha.

    Args:
        distances: Dystanse poszczególnych zdjęć ucha w bazie do nowo dodanego zdjęcia.

        class_labels: Etykiety opisujące do której klasy należy zdjęcie w *distances*.

    Returns:
        arr: Lista elementów zawierających w każdym wierszu:
            - klasę do której należą zdjęcia,
            - średnią odległość ucha z danej klasy do nowo dodanego ucha.
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

