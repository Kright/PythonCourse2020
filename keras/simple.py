import keras
import numpy as np
from keras.layers import InputLayer, Dense
from keras.models import Sequential


def f(arr: np.ndarray) -> np.ndarray:
    noise = np.random.normal(1.0, size=(arr.shape[0]))
    return arr[:, 0] * 2 - arr[:, 1] * 0.5 + noise


model = Sequential()
model.add(InputLayer(input_shape=(2,)))
model.add(Dense(1, activation=None, use_bias=False))

model.compile(loss=keras.losses.mse, optimizer=keras.optimizers.SGD(learning_rate=0.01), metrics=["mse"])

x_train = np.random.normal(0, 1.0, size=(1000, 2))
print(x_train.shape)

y_train = f(x_train)
print(y_train.shape)

x_test = np.array([[1, 0], [0, 1]])
y_test = f(x_test)

for _ in range(10):
    model.fit(x_train, y_train, shuffle=True, validation_data=(x_test, y_test))
    p = model.predict(x_test)
    print(p)
