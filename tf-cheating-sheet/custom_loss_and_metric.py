#!/usr/bin/env python
# coding: utf-8

# ---------------------------------------
# LOSS
'''
1.일반 계산만 수행하는 loss
2. 하이퍼파라미터를 사용하는 loss
'''
# ---------------------------------------

import tensorflow as tf
from tensorflow.keras import backend as K

# mse
def mse(y_true, y_pred):
    return tf.reduce_mean(tf.math.square(y_true - y_pred), axis = -1)

# binary_crossentropy
def binary_crossentropy(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    total_loss = -y_true * tf.math.log(y_pred) - (1.0 - y_true) * tf.math.log(1.0 - y_pred)
    
    return total_loss

# categorical_crossentropy
def categorical_crossentropy(y_true, y_pred):
    y_pred = y_pred / tf.reduce_sum(y_pred, axis = -1, keepdims = True)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    
    return -tf.reduce_sum(y_true * tf.math.log(y_pred), axis = -1)

# huber loss
def huber_loss(y_true, y_pred):
    threshold = 1
    error = y_true - y_pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)


model.compile(optimizer = 'adam', 
              loss = huber_loss,
              metrics = ['acc'])

# ---------------------------------------

# 파라미터 변경하고자하는 커스텀손실 - 1. nested 함수로 정의
def my_huber_loss_with_threshold(threshold):
    def my_huber_loss(y_true, y_pred):
        threshold = 1
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)
return my_huber_loss


from tensorflow.keras.losses import Loss

# 파라미터 변경하고자하는 커스텀손실 - 2. Loss 클래스 상속
class MyHuberLoss(Loss):
    def __init__(self, threshold = 1):
        super().__init__()
        self.threshold = threshold
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) <= self.threshold
        small_error_loss = tf.square(error) / 2
        big_error_loss = self.threshold * (tf.abs(error) - (0.5 * self.threshold))
        return tf.where(is_small_error, small_error_loss, big_error_loss)



model.compile(optimizer = 'adam', 
              loss = MyHuberLoss(2.0),
              metrics = ['acc'])


# ---------------------------------------
# METRIC
# ---------------------------------------
from tensorflow.keras import backend as K

def recall_metric(y_true, y_pred):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0.0, 1.0)))
    pred_pos = K.sum(K.round(K.clip(y_true, 0.0, 1.0)))
    recall = true_pos / (pred_pos + K.epsilon())
        
    return recall

def precision_metric(y_true, y_pred):
    true_pos = K.sum(K.round(K.clip(y_true * y_pred, 0.0, 1.0)))
    pred_pos = K.sum(K.round(K.clip(y_pred, 0.0, 1.0)))
    precision = true_pos / (pred_pos + K.epsilon())
        
    return precision

def f1_metric(y_true, y_pred):
    recall = recall_metric(y_true, y_pred)
    precision = precision_metric(y_true, y_pred)
    
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

model.compile(optimizer = 'adam', 
              loss = "mse",
              metrics = ['acc', f1_metric])

