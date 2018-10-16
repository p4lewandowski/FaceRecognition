from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense
from keras_vggface.vggface import VGGFace
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from FaceRecognition_imagepreprocessing import face_recording, take_image
from FaceRecognition_cnn_stream import OutputStream
import numpy as np
from keras import optimizers
import time, datetime

class Cnn_model:
    """
    Klasa umożliwiająca detekcję twarzy z użyciem konwolucyjnych sieci neuronowych. Zawiera wszystkie
    dane wejściowe, obliczeniowe i model sieci służący do obliczeń. Dodatkowo zawiera funkcje
    umożliwiające dodanie nowej osoby do bazy oraz identyfikację osób.
    """

    def __init__(self):
        # Prametry początkowe dla modelu
        self.label_id = []
        self.img_data = []
        self.index = 0
        # Ilość epok równa 50, ilość klas do określenia przy dodawaniu osób (poprzez kolejne dodane osoby)
        self.nb_class = 0
        self.epochs_num = 50
        self.fotosperclass_number = 60


    def initialize_networkmodel(self):
        """
        Funkcja inicjalizuje model sieci. Model użyty to VGGFace bez trzech ostatnich warstw, które
        zostały zamienione na warstwy o 64, 32 neuronach, z ostatnią warstwą o ilości neuronów
        równej ilości klas. Następuje inicjalizacja parametrów sieci takich jak momentum czy wskaźnik
        uczenia się. Określane są metody ewaluacji precyzji i straty.
        """

        ###### Struktura modelu ######
        vgg_model = VGGFace(include_top=False, weights='vggface', input_shape=(180, 180, 3))
        last_layer = vgg_model.get_layer('pool5').output
        # Add layers
        x = Flatten(name='flatten')(last_layer)
        x = Dense(64, activation='relu', name='fc6')(x)
        x = Dense(32, activation='relu', name='fc7')(x)
        out = Dense(self.nb_class, activation='softmax', name='fc8')(x)
        self.custom_vgg_model = Model(vgg_model.input, out)
        # Zamrożenie uczenia wszystkich warstw poza trzema ostatnimi, które zostały dodane
        layer_count = 0
        for layer in self.custom_vgg_model.layers:
            layer_count = layer_count + 1
        for l in range(layer_count - 3):
            self.custom_vgg_model.layers[l].trainable = False

        ###### Kompilacja modelu i parametrów uczenia (learning rate, momentum) ######
        sgd = optimizers.SGD(lr=5e-3, decay=1e-6, momentum=0.9, nesterov=True)
        self.custom_vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

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
        data = face_recording(gui=gui, im_count=self.fotosperclass_number, cnn=True)

        # Jeśli dodawana jest pierwsza twarz
        if not self.label_id:
            curr_id=0
        # Jeśli twarz nie jest pierwsza
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
            np.array(self.img_data), self.hot_label_id, test_size=0.2)
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
        validation_datagen = ImageDataGenerator()

        # Ustawienie parametru batchsize równego długości danych treningowych podzielonych przez 10 % długości
        self.train_batchsize = int(round(len(X_train)) / 10)

        self.train_generator = train_datagen.flow(
            X_train,
            y_train,
            shuffle=True,
            batch_size=self.train_batchsize)

        self.validation_generator = validation_datagen.flow(
            X_test,
            y_test,
            batch_size=self.train_batchsize)


    def train_cnn(self):
        """
        Funkcja odpowiada za trening konwolucyjnej sieci neuronowej. Do modelu przekazywane są
        zestawy zdjęć do uczenia i walidacji, a także następuje przekierowanie informacji nt. statusu
        uczenia sieci do oddzielnego strumienia. Umożliwia on wyświetlanie takiej informacji w
        aplikacji GUI. Dodatkowo dodany został Callback EarlyStopping, który w razie zmiany celności
        nie większej niż 1% - po 5 epokach bez zmiany skończy proces uczenia.
        """

        # Stworzenie listy Callbacków - jeden odpowiada za wyświetlanie danych w GUI - drugi za zatrzymanie uczenia
        # w przypadku gdy przez 5 epok accuracy nie urośnie o więcej niż 1%.
        self.stream_output = OutputStream()
        early_stopping = EarlyStopping(monitor='acc', min_delta=0.01, patience=5, verbose=0, mode='max')
        callback_list = [early_stopping, self.stream_output]

        # Historia rejestrująca zmiany w procesie uczenia
        self.history = self.custom_vgg_model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.train_generator.n / self.train_generator.batch_size,
            epochs=self.epochs_num,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.n / self.validation_generator.batch_size,
            verbose=1,
            callbacks=callback_list)

        # # Zapisz model do pliku
        # self.custom_vgg_model.save('trained_model_{}'.format(
        #     datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')))

    def recognize_face(self, gui=False):
        """
        Identyfikacja osoby z użyciem wcześniej wytrenowanej sieci neuronowej. Jeśli parametr *gui*
        jest dodany nastąpi wyświetlenie obrazu z kamery w aplikacji GUI, jeśli nie - pojawi się
        dodatkowe okno.

        Args:
            gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych
                z kamery w aplikacji.

        Returns:
            image: poszukiwana twarz (zdjęcie).
            model.predict: wartości prawdopodobieństwa przynależności do danej klasy.
        """
        try:
            assert self.custom_vgg_model
        except:
            raise AttributeError()

        image = take_image(gui=gui, cnn=True)
        return image, self.custom_vgg_model.predict(np.expand_dims(image, axis=0))[0]


    def accuracy_statistics(self):
        """
        Wyświetlenie danych dotyczących precyzji i straty w trakcie uczenia sieci neuronowej. Dane
        są podzielone na precyzję i stratę dla zestawu uczącego i zestawu walidacyjnego.
        """

        ##### Pokaż wyniki uczenia #####
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(len(acc))

        # Precyzja uczelnia
        plt.subplot(121)
        plt.plot(epochs, acc)
        plt.plot(epochs, val_acc)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # "Strata" uczenia
        plt.subplot(122)
        plt.plot(epochs,loss)
        plt.plot(epochs,val_loss)
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


