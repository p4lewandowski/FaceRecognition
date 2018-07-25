from FaceRecognition_eigenfaces import FaceRecognitionEigenfaces
from FaceRecognition_newfaces import EigenfaceRecognitionNewfaces
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


def test_accuracy(iterations=10, wavelet=False):

    precision = []
    f1 = []
    iter_range = range(0, iterations)

    for i in iter_range:

        fr_train = FaceRecognitionEigenfaces()
        fr_train.get_images(wavelet)

        result = []
        X_train, X_test, y_train, y_test = train_test_split(
            fr_train.image_matrix_flat.T, fr_train.labels, test_size=0.08)

        fr_train.image_matrix_flat = X_train.T
        fr_train.labels = y_train
        fr_train.image_count -= len(y_test)

        for i, x in enumerate(X_test):
            fr_train.get_eigenfaces()
            efr = EigenfaceRecognitionNewfaces(data=fr_train)
            x = np.matmul(x.T, fr_train.eigenfaces_flat.T)

            result.append(efr.recognize_face(x)[1])

        precision.append(accuracy_score(y_test, result))
        f1.append(f1_score(y_test, result, average='weighted'))

    fig, ax = plt.subplots()
    ax.plot(iter_range, precision,  label='Precision')
    ax.plot(iter_range, f1, label='F1 score')
    plt.axhline(np.mean(precision), label='Precision: {:.2f}'.format(np.mean(precision)), color='blue')
    plt.axhline(np.mean(f1), label='F1 score: {:.2f}'.format(np.mean(f1)), color='orange')
    legend = ax.legend(loc='lower right', shadow=True)
    plt.title('Precision and F1 score for multiple iterations')
    plt.xlabel('Iterations')
    plt.ylabel('Score')
    plt.show()


test_accuracy(iterations=10)
test_accuracy(iterations=10, wavelet=True)