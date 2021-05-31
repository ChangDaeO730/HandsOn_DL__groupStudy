#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------
# data preparing
# ---------------------------------------
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.0

# (28, 28) -> (28, 28, 1)
x_train = x_train[..., tf.newaxis]
from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                  test_size = 0.3, random_state = 777)
print(f'x_train shape: {x_train.shape} \nx_val shape: {x_val.shape}')


train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = train_ds.shuffle(1000).batch(32)
val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(32)

# ---------------------------------------
# Define Model
# ---------------------------------------
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras import Model

def get_model():
    inputs = Input(shape = (28, 28, 1))

    x = Conv2D(32, 3, activation = 'relu')(inputs)
    x = Flatten()(x)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(10, activation = 'softmax')(x)

    model = Model(inputs = inputs, outputs = x)

    return model

model = get_model()

# ---------------------------------------
# loss & optimizer 
# ---------------------------------------
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.optimizers import Adam

# model.compile하지 않고 별도로 정의
loss_object = SparseCategoricalCrossentropy()
optimizer = Adam()

# ---------------------------------------
# Metric 
# ---------------------------------------
from tensorflow.keras.metrics import Mean
from tensorflow.keras.metrics import SparseCategoricalAccuracy

train_loss = Mean(name='train_loss')
train_accuracy = SparseCategoricalAccuracy(name='train_accuracy')

val_loss = Mean(name='val_loss')
val_accuracy = SparseCategoricalAccuracy(name='val_accuracy')

# ---------------------------------------
# define train step
# ---------------------------------------
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        outputs = model(images, training=True)
        loss = loss_object(labels, outputs)

    # calculate gradient
    gradients = tape.gradient(loss, model.trainable_variables)
    # weight update
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    train_loss(loss)
    train_accuracy(labels, outputs)

@tf.function
def val_step(images, labels):
    outputs = model(images, training=False)
    v_loss = loss_object(labels, outputs)

    val_loss(v_loss)
    val_accuracy(labels, outputs)

# ---------------------------------------
# training
# ---------------------------------------
EPOCHS = 2

# epoch level loop
for epoch in range(EPOCHS):
    # 에폭마다 loss와 metric 초기화
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

    # train - step level loop
    for images, labels in train_ds:
        train_step(images, labels)

    # valid - step level loop
    for val_images, val_labels in val_ds:
        val_step(val_images, val_labels)

    print('Epoch: {}, train_loss: {}, train_acc: {} val_loss: {}, val_acc: {}'.format(
          epoch + 1,
          train_loss.result(), train_accuracy.result() * 100,
          val_loss.result(), val_accuracy.result() * 100))



