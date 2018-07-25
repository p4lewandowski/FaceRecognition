import numpy as np
import pywt
import cv2 as cv
import os
import matplotlib.pyplot as plt
from FaceRecognition_eigenfaces import FaceRecognitionEigenfaces
from FaceRecognition_newfaces import EigenfaceRecognitionNewfaces


fr = FaceRecognitionEigenfaces()
fr.get_images(wavelet=True)
fr.get_eigenfaces()
fr.show_me_things()
fr.stochastic_neighbour_embedding()

im = cv.imread('_newface.pgm', 0)
coeffs = pywt.dwt2(im, 'db1')
cA, (cH, cV, cD) = coeffs
plt.subplot(221)
plt.imshow(cA)
plt.subplot(222)
plt.imshow(cH)
plt.subplot(223)
plt.imshow(cV)
plt.subplot(224)
plt.imshow(cD)
plt.show()
print('done')


w = pywt.Wavelet('db1')
coeffs1 = pywt.WaveletPacket2D(im, w)

plt.subplot(211)
plt.imshow(coeffs1['a'].data)
plt.subplot(212)
plt.imshow(im)
plt.show()