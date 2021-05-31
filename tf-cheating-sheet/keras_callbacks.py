#!/usr/bin/env python
# coding: utf-8

# --------------------------------
# data preparing
# --------------------------------
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data(path='mnist.npz')
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                  test_size = 0.3, random_state = 777)
num_x_train = x_train.shape[0]
num_x_val = x_val.shape[0]
num_x_test = x_test.shape[0]

x_train = (x_train.reshape(-1, 28, 28, 1)) / 255
x_val = (x_val.reshape(-1, 28, 28, 1)) / 255
x_test = (x_test.reshape(-1, 28, 28, 1)) / 255

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

# --------------------------------
# model
# --------------------------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.layers import Input

inputs = Input(shape = (28, 28, 1))
x = Conv2D(32, (3, 3), activation = 'relu')(inputs)
x = Conv2D(32, (3, 3), activation = 'relu')(x)
x = MaxPooling2D(strides = 2)(x)
x = GlobalAveragePooling2D()(x)
x = Dense(10, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = x)
model.compile(optimizer = 'adam', 
              loss = 'categorical_crossentropy',
              metrics = ['acc'])



# --------------------------------
# use callbacks
# --------------------------------
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
import datetime

filepath = './best_model.hdf5'
logdir = './logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

callbacks = [ModelCheckpoint(filepath = filepath, 
                             monitor = 'val_loss', verbose = 1,
                             save_best_only = True),
             EarlyStopping(monitor = 'val_loss', patience = 3, verbose = 1),
             ReduceLROnPlateau(monitor = 'val_loss', patience = 3, factor = 0.2, 
                               verbose = 1, min_lr = 1e-5),
             TensorBoard(log_dir = logdir, histogram_freq = 1, 
                         write_graph = True, write_images = True)]


get_ipython().run_line_magic('load_ext', 'tensorboard')
# %reload_ext tensorboard

get_ipython().run_line_magic('tensorboard', '--logdir ./logs/fit')


model.fit(x_train, y_train,
         batch_size = 32,
         validation_data = (x_val, y_val),
         epochs = 10,
         callbacks = callbacks)

get_ipython().system('tensorboard dev upload --logdir ./logs/ --name "My test" --description "This is my first tensorboard"')

