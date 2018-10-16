import cv2 as cv
import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle


class FaceRecognitionEigenfaces():
    """
    Klasa do obliczeń związanych z twarzami własnymi, zawierająca dane dotyczące zdjęć i twarzy
    własnych, a także funkcje umożliwiające ich wyznaczenie.
    """

    def get_images(self):
        """
        Funkcja wczytuje zdjęcia twarzy z katalogu *imagedir* do pamięci. Następuje stworzenie
        macierzy zawierającej "spłaszone" zdjęcia (jako wektor), z których jest liczona "średnia
        twarz", która jest odejmowana od każdego ze zdjęć w macierzy. Inicjalizowane są parametry
        dotyczące oczekiwanej wariancji opisywanej przez składowe z PCA, zapisywana jest ilość
        wczytanych zdjęć, ich etykiet oraz rozmiar zdjęcia.
        """

        # Inicjalizacja parametrów początkowych
        self.image_count = 0
        self.labels = []
        self.rootdir = os.getcwd()
        self.datadir = os.path.join(self.rootdir, '..', 'Data')

        # Stworzenie pustej listy etykiet
        label_id = -1
        label_seen = []

        imagedir = os.path.join(self.datadir, 'detected_faces')

        # Przejście przez wszystkie zdjęcia w katalog, spłaszczenie i dopisanie do macierzy
        # *image_matrix* zawierającej wszystkie spłaszczone zdjęcia.
        image_matrix = []
        for file in os.listdir(imagedir):
            im = cv.imread(os.path.join(imagedir, file), 0)
            image_matrix.append(np.array(im).flatten())
            self.image_count += 1
            if not label_seen or int(file.split('_')[1].split('.')[0]) not in label_seen:
                label_id+=1
                label_seen.append(int(file.split('_')[1].split('.')[0]))
            self.labels.append(label_id)
        self.image_shape = im.shape[0]
        self.labels = np.array(self.labels)

        # Policzenie "twarzy średniej", poprzez średnią arytmetyczną każdego piksela we wszystkich
        # zdjęciach
        self.image_matrix_raw = np.array(np.transpose(image_matrix))

        self.mean_img = np.sum(self.image_matrix_raw, axis=1) / self.image_count
        self.mean_img = self.mean_img.reshape(self.image_shape, self.image_shape)
        # Subtract the mean from every flattened image
        self.image_matrix_flat = np.array(
            [x - self.mean_img.flatten() for x in self.image_matrix_raw.transpose()]).transpose()


    def get_eigenfaces(self, explained_variance=0.99):
        """
        Wyznaczenie twarzy własnych. W pierwszej kolejności wyznaczna jest macierz kowariancji z
        użyciem macierzy zawierającej wszystkie spłaszczone zdjęcia z twarzami z odjętą od nich
        "twarzą średnią". Jest ona obliczana jako A.T A w celu ograniczenia złożoności obliczeniowej.
        Wymiarowość takiej macierzy to macierz kwadratowa o długości/szerokości równej ilości
        dodanych zdjęć. Następnie liczone są wektory własne dla tej macierzy, czyli wyznaczane są
        twarze własne. Dodatkowo, funkcja odpowiada za reprezentację wszystkich wcześniej dodanych
        zdjęć do przestrzeni o nowej, mniejszej wymiarowości poprzez obliczenie wag odpowiadającym
        dodanym twarzom - tzn. zdjęcia twarzy rzutowane są na podprzestrzeń twarzy własnych czego
        wynikiem są wagi opisujące dane zdjęcia w nowej wymiarowości.
        """
        self.eigenfaces_n = explained_variance

        # Przygotuj macierz kowariancji równej A^T*A dla obliczeniowej wydajności
        cov_matrix = np.matmul(self.image_matrix_flat.transpose(), self.image_matrix_flat)
        cov_matrix /= self.image_count

        # Policz i wybierz wektory własne odpowiadające najwyższym wartościom własnym
        pca = PCA(n_components=self.eigenfaces_n)
        pca.fit(cov_matrix)

        # Lewostronne przemnożenie aby otrzymać właściwe wektory własne
        eigenvectors = np.matmul(self.image_matrix_flat, np.transpose(pca.components_))
        pca = PCA(n_components = self.eigenfaces_n)
        pca.fit(eigenvectors.transpose())
        self.eigenfaces_flat = pca.components_
        self.eigenfaces_n = len(pca.components_)

        # Policz wagi dla każdego zdjęcia - jest to reprezentacja twarzy w przestrzeni o zredukowanej
        # wymiarowosci
        self.face_weights = np.matmul(self.image_matrix_flat.transpose(), self.eigenfaces_flat.transpose())
        # Przekształć wektory własne z formy macierzy z wektorami do formy macierzy ze "zdjęciami"
        # twarzy własnych ilość_zdjęć X wymiar X wymiar2
        self.eigenfaces = np.array(self.eigenfaces_flat).reshape((self.eigenfaces_n, self.image_shape, self.image_shape))
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

    def stochastic_neighbour_embedding(self):
        """
        Redukcja wymiarowości danych w celu ich wizualizacji w przestrzeni o mniejszej wymiarowości
        używając metody t-SNE.
        """
        self.t_sne = TSNE(n_components=2, perplexity=5, early_exaggeration=16).fit_transform(self.face_weights)


    def transfer_image(self, image):
        """
        Funkcja rzutuje zdjęcie twarzy na podprzestrzeń twarzy własnych i zwraca wagi opisujące daną
        twarz.

        Args:
            image: Zdjęcie w formacie wektora wartości.

        Returns:
            image: Zdjęcie opisane w nowej przestrzeni.
        """

        # Odjęcie od zdjęcia twarzy średniej a następnie rzutowanie na twarze własne
        image = image - self.mean_img.flatten().transpose()
        image = np.matmul(image.T, self.eigenfaces_flat.T)
        return image

    def reconstruct_image(self, data, weights=False):
        """
        "Rekonstrukcja" twarzy na podstawie indeksu zdjęcia odnoszącego się do wag obliczonych po
        rzutowaniu bądź bezpośrednio na podstawie wektora wag.

        Args:
            data: Indeks zdjęcia do macierzy zawierającej wszystkie reprezentacje twarzy
                  w nowej przestrzeni bądź wektor wag opisujący twarz.
            weights: Jeśli *True* - *data* zawiera wektor wag opisujący twarz.
                     Jeśli *False* - *data* zawiera indeks zdjęcia w macierzy zawierajacej
                     wszystkie reprezentacje twarzy.

        Returns:
            reconstructed_face: Zwrócone zostaje zrekonstruowane zdjęcie.
        """

        reconstructed_face = np.copy(self.mean_img)
        if not weights:
            reconstructed_face += np.dot(self.face_weights[data], self.eigenfaces_flat) \
                .reshape(self.image_shape, self.image_shape)
        else:
            reconstructed_face += np.dot(data, self.eigenfaces_flat) \
                .reshape(self.image_shape, self.image_shape)

        return reconstructed_face

    def save_to_file(self):
        """
        Możliwość zapisu całego obiektu do pliku, wraz z obliczonymi wcześniej twarzami własnymi.
        """
        dbdir = os.path.join(self.datadir,'Database')
        pickle.dump(self, open("{}\\{}images-{}people.p".format(dbdir, self.image_count,
                                                               len(np.unique(self.labels))), "wb"))

def reconstruction_vs_explainedvariance():
    """
    Funkcja wyświetla wykres porównujący rekonstrukcję twarzy z użyciem różnych wartości oczekiwanej
    procentowej wariancji w trakcie obliczeń PCA. Prezentowane rekonstrukcje dla procentowej
    wariancji równej: 0.95, 0.99, 0.999, 0.9999 procent.
    Funkcja statyczna.
    """
    import matplotlib.pyplot as plt

    fr = FaceRecognitionEigenfaces()
    fr.get_images()

    fr.get_eigenfaces(explained_variance=0.95)
    im1 = fr.reconstruct_image(fr.face_weights[2], weights=True)
    fr.get_eigenfaces(explained_variance=0.99)
    im2 = fr.reconstruct_image(fr.face_weights[2], weights=True)
    fr.get_eigenfaces(explained_variance=0.999)
    im3 = fr.reconstruct_image(fr.face_weights[2], weights=True)
    fr.get_eigenfaces(explained_variance=0.9999)
    im4 = fr.reconstruct_image(fr.face_weights[2], weights=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.xticks(())
    plt.yticks(())
    plt.title("Oryginalna twarz", size=12)
    plt.suptitle("Porównanie rekonstrukcji twarzy zależne od procentowej wariancji przy obliczeaniu PCA")
    plt.imshow(fr.image_matrix_raw.T[2].reshape(86, 86), cmap=plt.cm.gray)

    plt.subplot(2, 3, 2)
    plt.xticks(())
    plt.yticks(())
    plt.title("0.95% wariancji", size=12)
    plt.imshow(im1, cmap=plt.cm.gray)
    plt.subplot(2, 3, 3)
    plt.xticks(())
    plt.yticks(())
    plt.title("0.99% wariancji", size=12)
    plt.imshow(im2, cmap=plt.cm.gray)
    plt.subplot(2, 3, 5)
    plt.xticks(())
    plt.yticks(())
    plt.title("0.999% wariancji", size=12)
    plt.imshow(im3, cmap=plt.cm.gray)
    plt.subplot(2, 3, 6)
    plt.xticks(())
    plt.yticks(())
    plt.title("0.9999% wariancji", size=12)
    plt.imshow(im4, cmap=plt.cm.gray)
