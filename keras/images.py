import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Activation
from keras.models import Sequential


def save_image(arr, name):
    if arr.shape[0] == 1:
        arr = arr[0]
    if len(arr.shape) > 2 and arr.shape[2] == 1:
        arr = np.reshape(arr, newshape=(arr.shape[0], arr.shape[1]))
    plt.imsave(name, arr)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train[:, :, :, np.newaxis] / 255.0
x_test = x_test[:, :, :, np.newaxis] / 255.0

print(x_train.shape)
print(y_train.shape)

model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 1)))

model.add(Conv2D(16, kernel_size=(3, 3), use_bias=False)) # 26
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Conv2D(16, kernel_size=(3, 3), use_bias=False)) # 24
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPool2D(2)) # 12

model.add(Conv2D(32, kernel_size=(3, 3), use_bias=False)) # 10
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(MaxPool2D(2)) # 5

model.add(Conv2D(64, kernel_size=(3, 3), use_bias=False)) # 3
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(10, activation="softmax"))



model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(learning_rate=0.01), metrics=["accuracy"])


model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
