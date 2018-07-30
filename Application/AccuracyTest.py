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

        # t = [ind for ind, x in enumerate(y_train)if x == 14]
        # for elem in t:
        #     plt.imshow(np.reshape(X_train[elem], (86, 86)))

        for i, x in enumerate(X_test):
            fr_train.get_eigenfaces()
            efr = EigenfaceRecognitionNewfaces(data=fr_train)
            conf, f_id, im, closest_f, closest_f_id = efr.recognize_face(direct_im=np.reshape(
                x, (fr_train.image_shape, fr_train.image_shape)))
            result.append(f_id)

            # plt.subplot(211)
            # plt.imshow(im, cmap=plt.cm.bone)
            # plt.subplot(212)
            # plt.imshow(closest_f, cmap=plt.cm.bone)
            # plt.show()

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


test_accuracy(iterations=40)
test_accuracy(iterations=40, wavelet=True)