#!/usr/bin/env python
# coding: utf-8

# -----------------------------------
# 1. Activation 함수에 직접 전달
# -----------------------------------
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, Activation
from tensorflow.keras.models import Model

# define my activation function
def Mish(x):
    return x * K.tanh(K.softplus(x))

inputs = Input(shape = (28, 28))

x = Flatten()(inputs)
x = Dense(50)(x)
x = Activation(Mish)(x) # feed to keras Activation Layer
x = Dense(30)(x)
x = Activation(Mish)(x)
x = Dense(10, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = x)


# -----------------------------------
# 2.1 커스텀 객체 목록을 사용
# -----------------------------------
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, Activation
from tensorflow.keras.models import Model

from tensorflow.keras.utils import get_custom_objects

# class define - Activation layer 상속
class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = 'Mish'

# activation 연산 정의
def mish(x):
    return x * K.tanh(K.softplus(x))

# global custom object list에 정의한 객체 전달하기
get_custom_objects().update({'mish': Mish(mish)})


# 문자열로 전달하여 사용가능
inputs = Input(shape = (28, 28))
x = Flatten()(inputs)
x = Dense(50)(x)
x = Activation('mish')(x)
x = Dense(30)(x)
x = Activation('mish')(x)
x = Dense(10, activation = 'softmax')(x)

model = Model(inputs = inputs, outputs = x)

# -----------------------------------
# 2.2 커스텀 객체 목록을 사용
# -----------------------------------

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, Activation
from tensorflow.keras.models import Model

from tensorflow.keras.utils import custom_object_scope

def Mish(x):
    return x * K.tanh(K.softplus(x))

# scope 지정하여 해당범위 내에서만 사용
with custom_object_scope({'mish':Mish}):
    inputs = Input(shape = (28, 28))
    
    x = Flatten()(inputs)
    x = Dense(50)(x)
    x = Activation('mish')(x)
    x = Dense(30)(x)
    x = Activation('mish')(x)
    x = Dense(10, activation = 'softmax')(x)

# 아래와 같이 루프 밖에서 실행시 에러
# x = Activation(‘mish’)(x) 

model = Model(inputs = inputs, outputs = x)


#  Tensorflow Addon 
# https://github.com/tensorflow/addons
# radam

import tensorflow_addons as tfa
model.compile(optimizer = tfa.optimizers.RectifiedAdam(), loss = 'mse')

