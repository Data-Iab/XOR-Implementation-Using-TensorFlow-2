import tensorflow as tf
from tensorflow import keras
import numpy as np
from pprint import pprint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

train_x = np.array([[1,1],
                    [1,0],
                    [0,1],
                    [0,0]],"float32")

train_y = np.array([[1],
                    [0],
                    [0],
                    [1]],"float32")

model = Sequential()
model.add(Dense(2,'sigmoid',input_dim=2))
model.add(Dense(1,'sigmoid'))
model.compile(loss = keras.losses.binary_crossentropy,optimizer=keras.optimizers.Adam(0.1))
model.fit(train_x,train_y,epochs= 1000)
print(model.predict(train_x))