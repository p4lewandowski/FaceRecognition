from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras_vggface.vggface import VGGFace
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers
import time, datetime
import os
import cv2 as cv
from sklearn.metrics import confusion_matrix
import keras

class Cnn_model:
    """
    Klasa umożliwiająca detekcję twarzy z użyciem konwolucyjnych sieci neuronowych. Zawiera wszystkie
    dane wejściowe, obliczeniowe i model sieci służący do obliczeń. Dodatkowo zawiera funkcje
    umożliwiające dodanie nowej osoby do bazy oraz identyfikację osób.
    """

    def __init__(self):
        # Prametry początkowe dla modelu
        self.label_id = []
        self.label_identities = []
        self.img_data = []
        self.index = 0
        # Ilość epok równa 20, ilość klas do określenia przy dodawaniu osób (poprzez kolejne dodane osoby)
        self.nb_class = 0
        self.epochs_num = 3000
        self.rootdir = os.getcwd()
        self.file_dir = os.path.join('Data') # data_processed_renamed # ear_database

    def load_data(self):

        for subdir, dirs, files in os.walk(self.file_dir):
            for file in files:

                # Przygotuj etykiety
                self.label_id.append([file.split('_')[1] ,file.split('_')[2].split('.')[0]])
                self.label_identities.append(int(file.split('_')[1]))

                # Zgromadz dane
                im = cv.imread(os.path.join(subdir, file), 0)
                self.img_data.append(np.expand_dims(im, axis=3))

        self.img_data = np.array(self.img_data)
        seen = []
        new_id = -1
        temp_label_id = []
        for element in self.label_id:
            if element not in seen:
                seen.append(element)
                new_id+=1
            temp_label_id.append(new_id)

        self.label_id = temp_label_id
        self.nb_class = len(set(self.label_id))
        print(self.label_id)
        print("Number of images:", len(self.label_identities))
        print(self.label_identities)

    def batch_preparation(self):
        """
        Funkcja generuje batche zdjęć i dzieli je na zestawy do uczenia i do walidacji. Zestaw danych
        do uczenia jest powiększony przez generowane dodatkowo zdjęcia o różnych przekształceniach
        (rotacja, przesunięcia, odwrócenie zdjęcia).
        """

        # Hot encoding - wymagana forma etykiet dla uczenia sieci
        self.hot_label_id = to_categorical(self.label_id)
        # Rozbicie danych na treningowe i do walidacji
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(self.img_data), self.hot_label_id, test_size=0.15)
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest')
        validation_datagen = ImageDataGenerator()

        self.batchsize = int(round(len(X_train)) / 10)

        self.train_generator = train_datagen.flow(
            X_train,
            y_train,
            shuffle=True,
            batch_size=self.batchsize)

        self.validation_generator = validation_datagen.flow(
            X_test,
            y_test,
            batch_size=self.batchsize)

    def initialize_cnnmodel(self):

        input_shape = np.shape(self.img_data[0])

        model = Sequential()
        # Block 1
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.1))
        # Block 2
        model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
        model.add(Dropout(0.2))
        # Block 3
        model.add(Conv2D(32, kernel_size=(2, 2), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Dropout(0.2))
        # Fully-connected Block
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.nb_class, activation='softmax'))

        ###### Kompilacja modelu i parametrów uczenia (learning rate, momentum) ######
        sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['top_k_categorical_accuracy'])
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        model.summary()

        self.model = model

    def train_cnn(self):
        """
        Funkcja odpowiada za trening konwolucyjnej sieci neuronowej. Do modelu przekazywane są
        zestawy zdjęć do uczenia i walidacji, a także następuje przekierowanie informacji nt. statusu
        uczenia sieci do oddzielnego strumienia. Umożliwia on wyświetlanie takiej informacji w
        aplikacji GUI. Dodatkowo dodany został Callback EarlyStopping, który w razie zmiany celności
        nie większej niż 1% - po 5 epokach bez zmiany skończy proces uczenia.
        """

        print(self.nb_class)
        # Historia rejestrująca zmiany w procesie uczenia
        self.history = self.model.fit_generator(
            self.train_generator,
            steps_per_epoch=self.train_generator.n / self.train_generator.batch_size,
            epochs=self.epochs_num,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.n / self.validation_generator.batch_size,
            verbose=1)

        out = self.model.predict(self.img_data)
        conf_matrix = confusion_matrix(self.label_id, np.argmax(out, axis=1))
        with open('conf_matrix.txt', 'w') as f:
            f.write(np.array2string(conf_matrix))

        # Zapisz model
        # self.model.save('trained_model_{}'.format(
        #      datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')))

    def accuracy_statistics(self, acc=True):
        """
        Wyświetlenie danych dotyczących precyzji i straty w trakcie uczenia sieci neuronowej. Dane
        są podzielone na precyzję i stratę dla zestawu uczącego i zestawu walidacyjnego.
        """

        ##### Pokaż wyniki uczenia #####
        if not acc:
            acc = self.history.history['top_k_categorical_accuracy']
            val_acc = self.history.history['val_top_k_categorical_accuracy']

        if acc:
            acc = self.history.history['acc']
            val_acc = self.history.history['val_acc']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']
        epochs = range(len(acc))

        # Precyzja uczenia
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

c = Cnn_model()
c.load_data()
c.batch_preparation()
c.initialize_cnnmodel()
c.train_cnn()
c.accuracy_statistics()