import cv2 as cv
import os
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
import numpy as np

# Parametry dla wykrywania twarzy metodą viola-jones: skala, ilość sąsiadów
scale_factor = 1.1
min_neighbors = 5
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


def image_selection():
    """
    W ścieżce *filedir* następuje wybór zdjęć, na których detekcja twarzy jest pozytywna. Nastepuje
    wycięcie obszaru zainteresowania za zdjęcia, przeskalowanie go do stałego rozmiaru a także
    normalizacja odcieni szarości. Następnie zdjęcia zapisane są do ścieżki *savepath* jako obrazy
    wejściowe do detekcji twarzy z użyciem twarzy własnych.
    """

    # Ustaw ścieżki: zdjęcia wejściowe, zdjęcia do zapisania po przetworzeniu z wykrytymi twarzami
    rootdir = os.getcwd()
    file_dir = os.path.join(rootdir, '..', 'Data', 'raw_faces_data')
    savepath = os.path.join(file_dir, '..', 'detected_faces')
    id = 100

    for subdir, dirs, files in os.walk(file_dir):

        for file in files:

            im = cv.imread(os.path.join(subdir, file), 0)
            faces = face_cascade.detectMultiScale(im, scale_factor, min_neighbors)

            for (x, y, w, h) in faces:
                crop_img = im[y:y + h, x:x + w]
                crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
                crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)
                cv.imwrite(os.path.join(savepath, '{}_{}.pgm'.format(id, subdir.split('\\s')[1])),
                           crop_img)
                id += 1

                # Plot the images with detected faces
                # cv.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # cv.imshow('img', im)
                # cv.waitKey(0)
                # cv.destroyAllWindows()

def detect_face(image):
    """
    Metoda umożliwia detekcję twarzy z wykorzystaniem metody Viola-Jones. Po detekcji następuje
    wyświetlenie zdjęcia z zaznaczonymi twarzami.

    Args:
        image: Zdjęcie na którym zostaną zaznaczone twarze.

    Returns:
        image: Zdjęcie z zaznaczonymi twarzami
    """
    faces = face_cascade.detectMultiScale(cv.cvtColor(image, cv.COLOR_BGR2GRAY), scale_factor, min_neighbors)
    for (x, y, w, h) in faces:
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

def image_cropping(im = None, cnn=False):
    """
    Funkcja umożliwia wycięcie obszaru zainteresowania ze zdjęcia, gdzie obszar zainteresowania
    zależny jest od użycia metody rozpoznawania twarzy: twarzy własnych, czy sieci neuronowych. W
    przypadku twarzy własnych zdjęcia zawierają jedynie rozpoznaną twarz. W przypadku sieci twarz i
    niewielka ramka dookoła twarzy jest wycinana do dalszej analizy. Wycięty obszar jest
    przeskalowywany i normalizowany.

    Args:
        im: Przekazane zdjęcie, na którym ma zostać dokonana detekcja twarzy.

        cnn: Jeśli *True* - detekcja twarzy dla sieci neuronowej, większy obszar zainteresowania
             niż tylko twarz jest wycinany. Wycięty obszar ze zdjęcia RGB. Jeśli *False* - tylko
             twarz jest wycinana. Wycięty obszar ze zdjęcia w skali szarości.

    Returns:
        crop_img: Wycięty obszar zainteresowania.
    """


    # Reprezentacja twarzy w skali szarości potrzebna (openCV używa domyślnie BGR)
    if cnn:
        im_init = bgr2rgb(im, qim=False)
        im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
        # Jeśli zdjęcie zostało poprawnie wykonane (nie same 0)
        if not np.any(im_init):
            return False

    faces = face_cascade.detectMultiScale(im, scale_factor, min_neighbors)
    # Jeśli wykryto twarz
    if np.any(faces):
        # Wybranie największej twarzy (najbliższej)
        (x, y, w, h) = faces[0]
        for i, coord in enumerate(faces):
            if coord[2] > w:
                (x, y, w, h) = coord

        max_rows, max_cols = im.shape
        # Dla cnn wycinane jest zdjęcie z dodatkowym marginesem o wielkości 15 pikseli, dla cnn zdjęcia
        # są przeskalowane do 180x180, dla twarzy własnych 86x86
        if cnn:
            bias = 15
            crop_img = im_init[max(0, y-bias):min(max_rows,y + h + bias),
                       max(0, x-bias):min(max_cols, x + w + bias)]
            crop_img = cv.resize(crop_img, dsize=(180, 180), interpolation=cv.INTER_CUBIC)
            crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)
            # Model do którego dołączone są ostatnie nowe warstwy był trenowany na zdjęciach z wartościami
            # od 0 do 1
            crop_img = crop_img / 255.

        else:
            crop_img = im[max(0,y):min(max_rows, y + h),
                       max(0, x):min(max_cols, x + w)]
            crop_img = cv.resize(crop_img, dsize=(86, 86), interpolation=cv.INTER_CUBIC)
            crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)

        return crop_img

    else:
        return False


def face_recording(gui=False, im_count=20, idle=True, cnn=False):
    """
    Zrobienie zdjęć, przekazywanych do bazy danych w celu dodania kolejnej osoby. Zdjęcia są
    przetworzone do potrzeb bazy danych. Pomiędzy każdym dodanym poprawnie zdjęciem umożliw kilka
    sekund bez obliczeń - przeciwdziałanie dodaniu zbyt wielu podobnych zdjęć do bazy.

    Args:
        gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych
        z kamery w aplikacji.

        im_count: Ilość zdjęć osoby do wykonania.

        idle: Ilość klatek bezczynności przed rozpoczęciem analizy obrazu.

        cnn: Jeśli *True* - operacje na zdjęciu kolorowym dla sieci neuronowej. Jeśli *False* -
        operacje na zdjęciu czarno białym dla twarzy własnych.

    Returns:
        face_data: Zdjęcia zebrane przez funkcję.
    """

    face_data = []
    # Ustaw 20 klatek wolnych bez obliczeń - czas "oczekiwania" (ale z wyświetleniem obrazu)
    # dla przygotowania użytkownika
    if idle:
        idle = 20

    while (len(face_data) < im_count):
        # Analiza klatek
        ret, frame = cap.read()
        drawing_frame = np.copy(frame)
        # Wykrywanie twarzy
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
        # Jeśli twarz wykryta
        for (x, y, w, h) in faces:
            # Wyświetl prostokąt
            cv.rectangle(drawing_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Jeśli czas "oczekiwania" równy 0 klatek
        if not idle:
            # Jeśli twarz wykryta
            if np.any(faces):
                # Tryb: cnn lub eigenface
                if not cnn:
                    im = image_cropping(im=gray)
                # Jeśli cnn
                if cnn:
                    im = image_cropping(im=frame, cnn=cnn)
                # Jeśli twarz wykryta i wycięta - dopisz zdjęcie do macierzy zdjęć i umożliw 3 klatki bez obliczeń
                if np.any(im):
                    face_data.append(im)
                    idle=2
        if idle:
            idle -= 1

        # Umożliw wizualizację obrazu
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Wyświetlenie podglądu z kamery w aplikacji używając obiektu GUI
        if gui:
            image = bgr2rgb(drawing_frame)
            gui.AddPersonLabel.setPixmap(QPixmap.fromImage(image))
        else:
            cv.imshow('frame', drawing_frame)

    cv.destroyAllWindows()

    if gui:
        gui.AddPersonLabel.clear()

    return face_data

def take_image(gui=False, idle=True, cnn=False):
    """
    Funkcja służąca do wykonania zdjęcia do identyfikacji osoby.

    Args:
        gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych z kamery
        w aplikacji.

        idle: Ilość klatek bezczynności przed rozpoczęciem analizy obrazu.

        cnn: Jeśli *True* - operacje na zdjęciu kolorowym dla sieci neuronowej. Jeśli *False* -
        operacje na zdjęciu czarno białym dla twarzy własnych.

    Returns:
        image: Wycięte zdjęcie z wykryta twarzą.
    """
    if idle:
        idle=20

    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if idle:
            idle -= 1

        if gui:
            image_gui = bgr2rgb(frame)
            gui.IdentifySearchLabel.setPixmap(QPixmap.fromImage(image_gui))
        else:
            cv.imshow('frame', frame)

        if not idle:
            # Wykryj twarze
            faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            # Jeśli wykryto
            for (x, y, w, h) in faces:
                if cnn:
                    image = image_cropping(im=frame, cnn=cnn)
                else:
                    image = image_cropping(im=gray)

                # Jeśli poprawnie przetworzono zdjęcie
                if image is not False:
                    cv.destroyAllWindows()
                    if gui:
                        gui.IdentifySearchLabel.clear()
                    return image

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def bgr2rgb(BGR_image, qim=True):
    """
    Konwersja zdjęcia w BGR (używanego przez openCV) do RGB.

    Args:
        BGR_image: Zdjęcie wejściowe BGR.
        qim: Jeśli *True* - stwórz QImage umożliwiające wyświetlenie obrazu w aplikacji PyQt. Jeśli
        *False* - pomiń ten krok.

    Returns:
        image: Zdjęcie RGB bądź Qimage ze zdjęcia RGB - zależnie od parametru *qim*.
    """
    RGB_image = np.array(cv.cvtColor(BGR_image, cv.COLOR_BGR2RGB))
    if not qim:
        return RGB_image
    image = QImage(
        RGB_image,
        RGB_image.shape[1],
        RGB_image.shape[0],
        RGB_image.shape[1] * 3,
        QImage.Format_RGB888
    )
    return image

def resize_image(image):
    """
    Funkcja skalująca zdjęcia aby zmieściły się one w aplikacji GUI.

    Args:
        image: zdjęcie w RGP do przeskalowania.

    Returns:
        image: przeslaowane zdjęcie.
    """
    ratio = np.shape(image)[1] / np.shape(image)[0]
    area = 850 * 450
    new_h = int(np.sqrt(area / ratio) + 0.5)
    new_w = int((new_h * ratio) + 0.5)
    image = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_CUBIC)

    return image


def bulk_face_visualization(path, app):
    """
    Zdjęcie wizualizujące wcześniej wczytane zdjęcie z pliku w aplikacji GUI. Wymagane jest
    przeskalowanie i zmiana z BGR na RGB oraz ustawienie centrowania.

    Args:
        path - ścieżka do pliku.
        app -  Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych z kamery
         w aplikacji.

    Returns:
    """
    image = resize_image(cv.imread(path))
    image_gui = bgr2rgb(image)
    app.GroupImageLabel.setPixmap(QPixmap.fromImage(image_gui))
    app.GroupImageLabel.setAlignment(QtCore.Qt.AlignCenter)

    return image

def bulk_face_detection(app):
    """
    Funkcja wykrywa twarze z wcześniej wczytanego zdjęcia i wyświetla je w aplikacji GUI poprzez
    zakreślenie niebieskim prostokątem.

    Args:
        app - Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych z kamery
         w aplikacji. Dodatkowo użyte jest app.bulk_image, które jest wcześniej wczytanym zdjęciem
         grupowym.

    """
    image = np.copy(app.bulk_image)
    image_marked = detect_face(image)
    app.GroupImageFoundLabel.setPixmap(QPixmap.fromImage(bgr2rgb(image_marked)))
    app.GroupImageFoundLabel.setAlignment(QtCore.Qt.AlignCenter)

def bulk_identify_faces(app, class_count):
    """
    Identyfikacja osób, dla których wykryto twarz w zdjęciu. Funkcja dostępna po wcześniejszym
    wczytaniu pliku zdjęciowego oraz wytrenowaniu sieci. Funkcja w pierwszej kolejności wycina
    wykryte twarze i sprawdza dla nich przynależność a następnie nadpisuje wykryte twarze twarzami
    reprezentacyjnymi (które są 5 wykonanym poprawnie zdjęciem przy dodawaniu osoby).

    Args:
         app - Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych z kamery
         w aplikacji. Dodatkowo użyte jest app.bulk_image, które jest wcześniej wczytanym zdjęciem
         grupowym.
    """
    bias = 15
    image_data = []
    # Wymiary zdjęcia grupowego
    max_rows, max_cols = cv.cvtColor(app.bulk_image, cv.COLOR_BGR2GRAY).shape
    # Wykrywanie twarzy
    faces = face_cascade.detectMultiScale(cv.cvtColor(app.bulk_image, cv.COLOR_BGR2GRAY),
                                          scale_factor, min_neighbors)
    # Wybór zdjęć reprezentacyjnych dla osób - jest to piąte wykonane zdjęcie dla każdej z osób
    representative_images = np.reshape([x*255 for x in app.cnn.img_data[4::class_count]],
                                       (int(len(app.cnn.img_data)/class_count), 180, 180, 3))
    representative_ids = app.cnn.label_id[4::class_count]

    # Identyfikacja z użyciem sieci neuronowej
    # Jeśli twarze wykryte
    if np.any(faces):
        for face in faces:
            (x, y, w, h) = face
            # Wytnij twarz i przetwórz tak jak w przypadku treningu sieci
            cropped_face = app.bulk_image[max(0, y - bias):min(max_rows, y + h + bias),
                           max(0, x - bias):min(max_cols, x + w + bias)]
            processed_face = cv.resize(cropped_face, dsize=(180, 180), interpolation=cv.INTER_CUBIC)
            processed_face = cv.normalize(processed_face, processed_face, 0, 255, cv.NORM_MINMAX)
            processed_face = processed_face / 255.
            predictions = app.cnn.custom_vgg_model.predict(np.expand_dims(processed_face, axis=0))[0]
            # Zapisz koordynaty twarzy, id zidentyfikowanej twarzy, wartosc procentową, warunek pewnej detekcji
            # Tylko gdy najbardziej prawdopodobna klasa - druga najbardziej prawdopodobna > 0.33
            image_data.append(((x, y, w, h),
                               np.argmax(predictions),
                               round(100*np.max(predictions), 1),
                               sorted(predictions)[-1] - sorted(predictions)[-2] > 0.33))

        # Nadpisz wykryte twarze zidentyfikowanymi twarzami
        image_detected = np.copy(app.bulk_image)
        image_labels = np.copy(app.bulk_image)
        for data in image_data:
            # Nadpisz zdjęciem - wcześniej przeskaluj
            image_to_place = cv.resize(representative_images[data[1]], dsize=(data[0][3], data[0][2]),
                                       interpolation=cv.INTER_CUBIC)
            image_to_place = cv.normalize(image_to_place, image_to_place, 0, 255, cv.NORM_MINMAX)
            image_detected[data[0][1]:data[0][1] + image_to_place.shape[0],
                data[0][0]:data[0][0] + image_to_place.shape[1]] = bgr2rgb(image_to_place.astype(np.uint8), False)
            # Dorysuj prostokąt
            cv.rectangle(image_detected, (data[0][0], data[0][1]),
                         (data[0][0] + data[0][2], data[0][1] + data[0][3]), (0, 0, 255), 2)
            # Dopisz wartości procentowe
            cv.putText(image_detected, '{}%'.format(data[2]), (data[0][0]+5, data[0][1]-5+data[0][3]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv.LINE_AA)

        # Nie nadpisuj, ale dodaj etykiety i wartości procentowe
        for data in image_data:
            # Jeśli pewne
            if data[3]:
                # Dopisz wartości procentowe
                cv.putText(image_labels, '{}%'.format(data[2]), (data[0][0]+15, data[0][1]-10+data[0][3]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv.LINE_AA)
                cv.putText(image_labels, 'ID: {}'.format(data[1]), (data[0][0]+15, data[0][1]-20+data[0][3]),
                           cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv.LINE_AA)

        app.GroupImageCoveredLabel.setPixmap(QPixmap.fromImage(bgr2rgb(image_detected)))
        app.GroupImageCoveredLabel.setAlignment(QtCore.Qt.AlignCenter)
        app.GroupImageIdentifiedLabel.setPixmap(QPixmap.fromImage(bgr2rgb(image_labels)))
        app.GroupImageIdentifiedLabel.setAlignment(QtCore.Qt.AlignCenter)

    else:
        app.GroupImageCoveredLabel.setPixmap(QPixmap.fromImage(bgr2rgb(app.bulk_image)))
        app.GroupImageCoveredLabel.setAlignment(QtCore.Qt.AlignCenter)
        app.GroupImageIdentifiedLabel.setPixmap(QPixmap.fromImage(bgr2rgb(app.bulk_image)))
        app.GroupImageIdentifiedLabel.setAlignment(QtCore.Qt.AlignCenter)

