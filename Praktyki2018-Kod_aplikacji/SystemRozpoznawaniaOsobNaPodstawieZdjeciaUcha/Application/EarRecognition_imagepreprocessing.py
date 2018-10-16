import cv2 as cv
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import QtCore
import numpy as np

# Parametry dla wykrywania ucha metodą viola-jones: skala, ilość sąsiadów
left_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_leftear.xml')
right_ear_cascade = cv.CascadeClassifier('haarcascade_mcs_rightear.xml')
scale_factor = 1.1
min_neighbors = 4

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

def image_cropping(im = None, cnn=False, ear='both'):
    """
    Funkcja umożliwia wycięcie obszaru zainteresowania ze zdjęcia, gdzie obszar zainteresowania
    zależny jest od użycia metody rozpoznawania ucha: uszu własnych, czy sieci neuronowych. Obszar
    zainteresowania to ucho i niewielka ramka dookoła ucha o stałej szerokości: 10 i 5 pikseli dla
    sieci neuronowych i uszu własnych. Wycięty obszar jest przeskalowywany, filtrowany z użyciem
    bilateral filter i normalizowany. Aby usprawnić działanie aplikacji przy wycinaniu ucha mogą
    być brane pod uwagę ucho lewe i prawo, bądź każde z nich osobno.

    Args:
        im: Przekazane zdjęcie, na którym ma zostać dokonana detekcja ucha.

        cnn: Jeśli *True* - detekcja ucha dla sieci neuronowej z szerokością ramki równą 10 pikseli.
        Jeśli *False* -  detekcja ucha dla uszu własnych z szerokością ramki równą 5 pikseli.

        ear: Opcja determinująca, które ucho ma być wykrywane i wycięte, do wyboru:
            - left
            - right
            - both.

    Returns:
        crop_img: Wycięty obszar zainteresowania o stałych wymiarach 62x100.
    """
    if ear=='left':
        ear_detected = left_ear_cascade.detectMultiScale(im, scale_factor, min_neighbors)
    elif ear=='right':
        ear_detected = right_ear_cascade.detectMultiScale(im, scale_factor, min_neighbors)
    elif ear=='both':
        # W przypadku obu uszu - scal wyniki z dwóch detektorów i wybierz największe - najbliższe ucho
        lear = left_ear_cascade.detectMultiScale(im, scale_factor, min_neighbors)
        rear = right_ear_cascade.detectMultiScale(im, scale_factor, min_neighbors)
        if np.any(lear) and np.any(rear):
            ear_detected = np.vstack((lear, rear))
        elif np.any(lear):
            ear_detected = lear
        elif np.any(rear):
            ear_detected = rear

    # Sprawdź czy detekcja poprawna
    if np.any(ear_detected):
        isPresent = True
    else:
        isPresent = False

    if(isPresent):
        # Wybierz największe ucho - najbliższe
        if np.any(ear_detected):
            (x, y, w, h) = ear_detected[0]
            for i, coord in enumerate(ear_detected):
                if coord[2] > w:
                    (x, y, w, h) = coord

        max_rows, max_cols = im.shape
        # Dla cnn wycinane jest zdjęcie z dodatkowym marginesem o wielkości 10 pikseli
        # Zdjęcia są przeskalowane do rozdzielczości 62x100
        dsize = (62, 100)
        if cnn:
            bias = 10
        else:
            bias = 5

        crop_img = im[max(0,y):min(max_rows, y + h),
                   max(0, x):min(max_cols, x + w)]
        crop_img = cv.resize(crop_img, dsize=dsize, interpolation=cv.INTER_CUBIC)
        crop_img = cv.bilateralFilter(crop_img, 6, 80, 80)
        crop_img = cv.normalize(crop_img, crop_img, 0, 255, cv.NORM_MINMAX)

        return crop_img

    else:
        return False


def ear_recording(gui=False, im_count=20, idle=True, cnn=False):
    """
    Zrobienie zdjęć, przekazywanych do bazy danych w celu dodania kolejnej osoby. Zdjęcia są
    przetworzone do potrzeb bazy danych. W pierwszych 40 klatkach wykonywana jest kalibracja -
    liczona jest ilość wystąpień ucha prawego i lewego (na korzyść największego ucha). Jeśli różnica
    jest większa niż 5 aplikacja zacznie filtrować obraz jedynie w poszukiwaniu ucha z większą liczbą
    wystąpień (prawego bądź lewego).

    Args:
        gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych
        z kamery w aplikacji.

        im_count: Ilość zdjęć osoby do wykonania.

        idle: Ilość klatek bezczynności przed rozpoczęciem analizy obrazu.

        cnn: Jeśli *True* - operacje na zdjęciu kolorowym dla sieci neuronowej. Jeśli *False* -
        operacje na zdjęciu czarno białym dla uszu własnych.

    Returns:
        ear_data: Zdjęcia zebrane przez funkcję.
    """

    ear_data = []
    right_ear_num = 0
    left_ear_num = 0
    # Wartość bool oznaczająca które ucho będzie poszukiwane
    isleftear = None
    # Ustaw 20 klatek wolnych bez obliczeń - czas "oczekiwania"
    # (ale z wyświetleniem obrazu) dla przygotowania użytkownika
    if idle:
        idle = 40

    while (len(ear_data) < im_count):
        # Analiza klatek
        ret, frame = cap.read()
        drawing_frame = np.copy(frame)

        # Wykrywanie ucha
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        if isleftear != True or isleftear == None:
            r_ear = right_ear_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            for (x, y, w, h) in r_ear:
                cv.rectangle(drawing_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        if isleftear == True or isleftear == None:
            l_ear = left_ear_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            for (x, y, w, h) in l_ear:
                cv.rectangle(drawing_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if idle>0:
            idle -= 1

        # Sprawdzenie czy można określić, które ucho ma być filtrowane
        if not idle and isleftear == None:
            if abs(right_ear_num - left_ear_num)>5:
                if left_ear_num > right_ear_num:
                    isleftear = True
                else:
                    isleftear = False
            # Jeśli nie sprawdzaj dalej
            else:
                isleftear = None
                idle+=1

        if not idle:
            # jeśli jakieś ucho wykryte
            if isleftear:
                if np.any(l_ear):
                    im = image_cropping(im=gray, cnn=cnn, ear='left')
            if not isleftear:
                if np.any(r_ear):
                    im = image_cropping(im=gray, cnn=cnn, ear='right')

            # Jeśli ucho wykryte i wycięte dopisz do macierzy zdjęć
            # Warunek try w przypadku gdy ostatnie zdjecie przed kalibracja nie zawieralo ucha
            try:
                if np.any(im):
                    ear_data.append(im)
            except UnboundLocalError:
                None

        if idle:
            # Liczenie wystąpień uszu
            if len(r_ear)>0 and len(l_ear) == 0:
                right_ear_num +=1
            elif len(l_ear)>0 and len(r_ear) == 0:
                left_ear_num +=1
            elif len(r_ear) == 0 and len(l_ear) == 0:
                None
            else:
                # Dodanie wystąpienia dla największego ucha
                (xr, yr, wr, hr) = r_ear[0]
                for i, coord in enumerate(r_ear):
                    if coord[2] > wr:
                        (xr, yr, wr, hr) = coord
                (xl, yl, wl, hl) = l_ear[0]
                for i, coord in enumerate(l_ear):
                    if coord[2] > wl:
                        (xl, yl, wl, hl) = coord
                if wl+hl > wr+hr:
                    left_ear_num += 1
                elif wl+hl < wr+hr:
                    right_ear_num += 1

        # Umożliw wizualizację obrazu
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        # Wyświetl podgląd z kamery w aplikacji używając obiektu GUI
        if gui:
            image = bgr2rgb(drawing_frame)
            gui.AddPersonLabel.setPixmap(QPixmap.fromImage(image))
        else:
            cv.imshow('frame', drawing_frame)

    cv.destroyAllWindows()

    if gui:
        gui.AddPersonLabel.clear()

    return ear_data

def take_image(gui=False, idle=True, cnn=False):
    """
    Funkcja służąca do wykonania zdjęcia do identyfikacji osoby.

    Args:
        gui: Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych z kamery
        w aplikacji.

        idle: Ilość klatek bezczynności przed rozpoczęciem analizy obrazu.

        cnn: Jeśli *True* - operacje na zdjęciu kolorowym dla sieci neuronowej. Jeśli *False* -
        operacje na zdjęciu czarno białym dla uszu własnych.

    Returns:
        image: Wycięte zdjęcie z wykrytym uchem.
    """
    if idle:
        idle=20

    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        isPresent = False

        if idle:
            idle -= 1

        if gui:
            image_gui = bgr2rgb(frame)
            gui.IdentifySearchLabel.setPixmap(QPixmap.fromImage(image_gui))
        else:
            cv.imshow('frame', frame)

        if not idle:
            # Wykryj ucho
            r_ear = right_ear_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            l_ear = left_ear_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
            # Jeśli znaleziono
            for (x, y, w, h) in r_ear:
                isPresent=True
            for (x, y, w, h) in l_ear:
                isPresent = True

            if isPresent:
                image = image_cropping(im=gray, cnn=cnn, ear='both')
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

def detect_ear(image):
    """
    Metoda umożliwia detekcję uszu z wykorzystaniem metody Viola-Jones. Po detekcji następuje
    wyświetlenie zdjęcia z zaznaczonymi uszami.

    Args:
        image: Zdjęcie na którym zostaną zaznaczone uszy.

    Returns:
        image: Zdjęcie z zaznaczonymi uszami - jeśli takowe istnieją.
    """
    lear = left_ear_cascade.detectMultiScale(cv.cvtColor(image, cv.COLOR_BGR2GRAY), scale_factor, min_neighbors)
    rear = right_ear_cascade.detectMultiScale(cv.cvtColor(image, cv.COLOR_BGR2GRAY), scale_factor, min_neighbors)

    if np.any(lear) and np.any(rear):
        ears = np.vstack((lear, rear))
    elif np.any(lear):
        ears = lear
    elif np.any(rear):
        ears = rear
    else:
        ears = []

    for (x, y, w, h) in ears:
        cv.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image

def bulk_ear_visualization(path, app):
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

def bulk_ear_detection(app):
    """
    Funkcja wykrywa uszy z wcześniej wczytanego zdjęcia i wyświetla je w aplikacji GUI poprzez
    zakreślenie niebieskim prostokątem.

    Args:
        app - Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych z kamery
         w aplikacji. Dodatkowo użyte jest app.bulk_image, które jest wcześniej wczytanym zdjęciem
         grupowym.

    """
    image = np.copy(app.bulk_image)
    image_marked = detect_ear(image)
    app.GroupImageFoundLabel.setPixmap(QPixmap.fromImage(bgr2rgb(image_marked)))
    app.GroupImageFoundLabel.setAlignment(QtCore.Qt.AlignCenter)

def bulk_identify_ears(app, class_count):
    """
    Identyfikacja osób, dla których wykryto uszy w zdjęciu. Funkcja dostępna po wcześniejszym
    wczytaniu pliku zdjęciowego oraz wytrenowaniu sieci. Funkcja w pierwszej kolejności wycina
    wykryte uszy i sprawdza dla nich przynależność a następnie nadpisuje wykryte uszy uszami
    reprezentacyjnymi (które są 5 wykonanym poprawnie zdjęciem przy dodawaniu osoby).

    Args:
         app - Obiekt głównej aplikacji GUI, dzięki któremu możliwa jest wizualizacja danych z kamery
         w aplikacji. Dodatkowo użyte jest app.bulk_image, które jest wcześniej wczytanym zdjęciem
         grupowym.
         class_count - ilość zdjęć wykonywanych dla jednej klasy w trakcie jej dodawania.
    """
    bias = 15
    image_data = []
    # Wymiary zdjęcia grupowego
    max_rows, max_cols = cv.cvtColor(app.bulk_image, cv.COLOR_BGR2GRAY).shape
    # Wykrywanie uszu
    lear = left_ear_cascade.detectMultiScale(cv.cvtColor(app.bulk_image, cv.COLOR_BGR2GRAY), scale_factor, min_neighbors)
    rear = right_ear_cascade.detectMultiScale(cv.cvtColor(app.bulk_image, cv.COLOR_BGR2GRAY), scale_factor, min_neighbors)
    if np.any(lear) and np.any(rear):
        ears = np.vstack((lear, rear))
    elif np.any(lear):
        ears = lear
    elif np.any(rear):
        ears = rear
    else:
        ears = []

    # Wybór zdjęć reprezentacyjnych dla osób - jest to piąte wykonane zdjęcie dla każdej z osób, zdjęcia 62x100
    representative_images = np.reshape([x*255 for x in app.cnn.img_data[4::class_count]],
                                       (int(len(app.cnn.img_data)/class_count), 100, 62))

    # Identyfikacja z użyciem sieci neuronowej
    # Jeśli uszy wykryte
    if np.any(ears):
        for ear in ears:
            (x, y, w, h) = ear
            # Wytnij ucho i przetwórz tak jak w przypadku treningu sieci
            cropped_face = app.bulk_image[max(0, y - bias):min(max_rows, y + h + bias),
                           max(0, x - bias):min(max_cols, x + w + bias)]
            processed_face = cv.resize(cropped_face, dsize=(62, 100), interpolation=cv.INTER_CUBIC)
            processed_face = cv.normalize(processed_face, processed_face, 0, 255, cv.NORM_MINMAX)
            processed_face = cv.cvtColor(processed_face, cv.COLOR_BGR2GRAY)
            predictions = app.cnn.model.predict(
                np.expand_dims(np.expand_dims(processed_face, axis=0), axis=4))[0]
            # Zapisz koordynaty ucha, id zidentyfikowanego ucha, wartosc procentową, warunek pewnej detekcji
            # Tylko gdy najbardziej prawdopodobna klasa - druga najbardziej prawdopodobna > 0.33
            image_data.append(((x, y, w, h),
                               np.argmax(predictions),
                               round(100*np.max(predictions), 1),
                               sorted(predictions)[-1] - sorted(predictions)[-2] > 0.33))

        # Nadpisz wykryte uszy zidentyfikowanymi uszami
        image_detected = np.copy(app.bulk_image)
        image_labels = np.copy(app.bulk_image)
        for data in image_data:
            # Nadpisz zdjęciem - wcześniej przeskaluj + inwersja
            image_to_place = cv.resize(representative_images[data[1]], dsize=(data[0][2], data[0][3]),
                                       interpolation=cv.INTER_CUBIC)
            image_to_place = cv.normalize(255-image_to_place, image_to_place, 0, 255, cv.NORM_MINMAX)
            image_to_place = cv.cvtColor(image_to_place,cv.COLOR_GRAY2RGB)
            # Nakładanie uszu
            image_detected[data[0][1]:data[0][1] + image_to_place.shape[0],
                data[0][0]:data[0][0] + image_to_place.shape[1]] = image_to_place.astype(np.uint8)
            # Dorysuj prostokąt
            cv.rectangle(image_detected, (data[0][0], data[0][1]),
                         (data[0][0] + data[0][2], data[0][1] + data[0][3]), (0, 0, 255), 2)
            # Dopisz wartości procentowe
            cv.putText(image_detected, '{}%'.format(data[2]), (data[0][0]+5, data[0][1]-5+data[0][3]),
                       cv.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1, cv.LINE_AA)

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

    # Jeśli nie zostały wykryte usze
    else:
        app.GroupImageCoveredLabel.setPixmap(QPixmap.fromImage(bgr2rgb(app.bulk_image)))
        app.GroupImageCoveredLabel.setAlignment(QtCore.Qt.AlignCenter)
        app.GroupImageIdentifiedLabel.setPixmap(QPixmap.fromImage(bgr2rgb(app.bulk_image)))
        app.GroupImageIdentifiedLabel.setAlignment(QtCore.Qt.AlignCenter)


