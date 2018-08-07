from keras.engine import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from keras import models
import cv2 as cv
import numpy as np
from keras import optimizers
import os
import time

def new_img_convert(img, h = 10, w = 10):
    img = img[h:-h, w:-w]
    img = cv.resize(img, (180, 180), interpolation=cv.INTER_CUBIC)
    img = img.astype('float32')
    img = img / 255.
    return img

# Start measuring time
start = time.time()
label_id = []
label_names = []
img_data = []
index = 0
# Change the batchsize according to RAM amount
train_batchsize = 64
val_batchsize = 5
# Custom parameters
nb_class = 0
hidden_dim = 512
epochs_num = 20


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

label = to_categorical(label_id)
print(np.shape(img_data), nb_class, np.shape(label), np.shape(label_names))

###### Define model ######
vgg_model = VGGFace(include_top=False, weights='vggface', input_shape=(180, 180, 3))
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

###### Prepare the data ######
X_train, X_test, y_train, y_test = train_test_split(np.array(img_data), label, test_size=0.1, random_state=42)
train_datagen = ImageDataGenerator(
      rotation_range=20,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train,
    y_train,
    shuffle=True,
    batch_size=train_batchsize)

validation_generator = validation_datagen.flow(
    X_test,
    y_test,
    batch_size=val_batchsize)

###### Finalize the model ######
sgd = optimizers.SGD(lr=5e-3, decay=1e-6, momentum=0.9, nesterov=True)
custom_vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
custom_vgg_model.save('my_model_pretrained_night')

###### Training ######
# Train the model
history = custom_vgg_model.fit_generator(
      train_generator,
      steps_per_epoch=train_generator.n/train_generator.batch_size,
      epochs=epochs_num,
      validation_data=validation_generator,
      validation_steps=validation_generator.n/validation_generator.batch_size,
      verbose=1)

custom_vgg_model.save('my_model_trained_night')

##### Show results #####
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

print("Whole process completed in time: {}s".format(time.time() - start))