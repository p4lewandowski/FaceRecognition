from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from EarRecognition_imagepreprocessing import ear_recording, take_image
from EarRecognition_cnn_stream import OutputStream
import numpy as np
from keras import optimizers
import time, datetime
from keras.models import load_model
import cv2 as cv


class Cnn_model:
    """
    Klasa umożliwiająca identyfikację ucha z użyciem konwolucyjnych sieci neuronowych. Zawiera
    wszystkie dane wejściowe, obliczeniowe i model sieci służący do obliczeń. Dodatkowo zawiera
    funkcje umożliwiające dodanie nowej osoby do bazy oraz identyfikację osób.
    """
    def __init__(self):
        # Prametry początkowe dla modelu
        self.label_id = []
        self.img_data = []
        self.index = 0
        self.fotosperclass_number = 50
        # Ilość epok równa 50, ilość klas do określenia przy dodawaniu osób (poprzez kolejne dodane osoby)
        self.nb_class = 0
        self.epochs_num = 50

    def model_compile(self):
        # Wczytanie przetrenowanego modelu i usunięcie ostatnich warstw pełnych i softmax
        model = load_model('Chosen_one')
        self.model = model
        self.model.pop()
        self.model.pop()
        self.model.pop()
        self.model.pop()
        # Dodanie nowych warstw
        self.model.add(Dense(64, activation='relu', name='fc6'))
        self.model.add(Dense(32, activation='relu', name='fc7'))
        self.model.add(Dense(self.nb_class, activation='softmax', name='fc8'))
        layer_count = 0
        for layer in self.model.layers:
            layer_count = layer_count + 1
        for l in range(layer_count - 3):
            self.model.layers[l].trainable = False
        # Parametry uczenia
        sgd = optimizers.SGD(lr=0.005, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    def add_person(self, **kwargs):
        """
        Funkcja umożliwia dodanie osoby do bazy do treningu sieci neuronowej. Jeśli parametr *gui*
        jest przekazany wizualizacja danych z kamery nastapi w aplikacji GUI. W innym wypadku
        wyświetlone zostanie okno z podglądem kamery.

        Args:
            **kwargs:
                gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych
                z kamery w aplikacji.
        """
        # Sprawdzenie czy parametr został przekazany
        gui = kwargs.get('gui', False)
        # Przekazanie parametru gui do kolejnej funkcji, gdzie jeden z elementów gui zostanie wykorzystany
        data = ear_recording(gui=gui, im_count=self.fotosperclass_number, cnn=True)

        # Jeśli dodawane jest pierwsze ucho
        if not self.label_id:
            curr_id=0
        # Jeśli kolejne
        else:
            curr_id = max(self.label_id)+1
        self.label_id.extend([curr_id]*len(data))
        self.img_data.extend(data)
        self.nb_class += 1

    def data_processing(self):
        """
        Funkcja generuje batche zdjęć i dzieli je na zestawy do uczenia i do walidacji. Zestaw danych
        do uczenia jest powiększony przez generowane dodatkowo zdjęcia o różnych przekształceniach
        (rotacja, przesunięcia, odwrócenie zdjęcia).
        """

        # Hot encoding - wymagana forma etykiet dla uczenia sieci
        self.hot_label_id = to_categorical(self.label_id)
        # Rozbicie danych na treningowe i do walidacji
        X_train, X_test, y_train, y_test = train_test_split(
            np.expand_dims(np.array(self.img_data), axis=4), self.hot_label_id, test_size=0.2)
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=False,
            fill_mode='nearest')
        validation_datagen = ImageDataGenerator()

        # Ustawienie parametru batchsize równego długości danych treningowych podzielonych przez 10 % długości
        self.batch_size = int(round(len(X_train)) / 10)

        self.train_generator = train_datagen.flow(
            X_train,
            y_train,
            shuffle=True,
            batch_size=self.batch_size)

        self.validation_generator = validation_datagen.flow(
            X_test,
            y_test,
            batch_size=self.batch_size)

    def train_cnn(self):
        """
        Funkcja odpowiada za trening konwolucyjnej sieci neuronowej. Do modelu przekazywane są
        zestawy zdjęć do uczenia i walidacji, a także następuje przekierowanie informacji nt. statusu
        uczenia sieci do oddzielnego strumienia. Umożliwia on wyświetlanie takiej informacji w
        aplikacji GUI. Dodatkowo dodany został Callback EarlyStopping, który w razie zmiany celności
        nie większej niż 1% - po 5 epokach bez zmiany skończy proces uczenia.
        """

        # Stworzenie listy Callbacków - jeden odpowiada za wyświetlanie danych w GUI - drugi za zatrzymanie uczenia
        # w przypadku gdy przez 5 epok acc nie urośnie o więcej niż 1%.
        self.stream_output = OutputStream()
        early_stopping = EarlyStopping(monitor='acc', min_delta=0.01, patience=5, verbose=0, mode='max')
        callback_list = [early_stopping, self.stream_output]

        # Historia rejestrująca zmiany w procesie uczenia
        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.train_generator.n / self.train_generator.batch_size,
            epochs=self.epochs_num,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.n / self.validation_generator.batch_size,
            callbacks=callback_list,
            verbose=2
        )

    def recognize_ear(self, gui=False):
        """
        Identyfikacja osoby z użyciem wcześniej wytrenowanej sieci neuronowej. Jeśli parametr *gui*
        jest dodany nastąpi wyświetlenie obrazu z kamery w aplikacji GUI, jeśli nie - pojawi się
        dodatkowe okno.

        Args:
            gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych
                z kamery w aplikacji.

        Returns:
            image: poszukiwane ucho (zdjęcie)
            model.predict: wartości prawdopodobieństwa przynależności do danej klasy
        """
        image = take_image(gui=gui, cnn=True)
        return image, self.model.predict(np.expand_dims(np.expand_dims(image, axis=0), axis=4))[0]
