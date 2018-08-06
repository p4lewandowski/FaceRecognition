from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras.utils import to_categorical

from keras import models
import cv2 as cv
import numpy as np
from keras import optimizers
import os
import time

def new_img_convert(img):
    img = cv.resize(img, (224, 224), interpolation=cv.INTER_CUBIC)
    img = im.astype('float32')
    img/=255
    # img=np.expand_dims(crop_img, axis=0)
    return img

# Start measuring time
start = time.time()
label_id = []
label_names = []
img_data = []
index = 0
# Custom parameters
nb_class = 0
hidden_dim = 128

for subdir, dirs, files in os.walk(os.path.join(os.getcwd(), 'filtered_dataset')):

    # Start indexing when something appears
    if not label_id:
        index = 0
    else:
        index +=1

    for file in files:
        im = cv.imread(os.path.join(subdir, file))
        img_data.append(new_img_convert(im))
        label_id.append(index)
        if ''.join(file.split('.')[0].split('_')[:-1]) not in label_names:
            label_names.append(''.join(file.split('.')[0].split('_')[:-1]))
            nb_class += 1

label_names = to_categorical(label_id)

print(np.shape(img_data), np.shape(label_names), nb_class)


# Define model
vgg_model = VGGFace(include_top=False, weights='vggface', input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, out)
# Turn some layers as not trainable | Last three dense layers are trainable
layer_count = 0
for layer in custom_vgg_model.layers:
	layer_count = layer_count+1
for l in range(layer_count-3):
	custom_vgg_model.layers[l].trainable=False
# Print summary of the model
# custom_vgg_model.summary()

sgd = optimizers.SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
custom_vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
custom_vgg_model.save('my_model')
# models.load_model()

print(time.time() - start)