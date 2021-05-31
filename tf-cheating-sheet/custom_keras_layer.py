#!/usr/bin/env python
# coding: utf-8

'''
1. lambda : 학습가능한 가중치를 사용하지 않는 커스텀층
2. Layer상속 
'''

# -----------------------------------------
# 1. Lambda layer
# -----------------------------------------

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K

# define any function
def custom_f(x):
    x_mean = K.mean(x)
    x_std = K.std(x)

    return (x - x_mean) / x_std

inputs = Input(shape = (5, ))
# Lambda로 함수를 wrapping하여 layer로 만들어줌
x = Lambda(custom_f)(inputs).
# x = Dense(32, activation = 'relu')(x)

model = Model(inputs, x)



# -----------------------------------------
# 2. Custom layer
'''
* build(input_shape) : 학습할 가중치 정의
* call(x) : 해당 층에서 수행할 핵심 연산 정의
* compute_output_shape(input_shape) : 출력 shape 정의
'''
# -----------------------------------------

from tensorflow.keras.layers import Layer, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.activations import relu

class CustomLayer(Layer):
    def __init__(self, num_hidden):
        super(CustomLayer, self).__init__()
        self.num_hidden = num_hidden
    
    # define weights
    def build(self, input_shape):
        self.kernels = self.add_weight('kernels',
                                       shape = [int(input_shape[-1]), 
                                                self.num_hidden])
        self.bias = self.add_weight('bias',shape = [self.num_hidden])
    
    # define layer calculations
    def call(self, x):
        return relu(tf.matmul(x, self.kernels) + self.bias)
    
    # define output shape
    def compute_output_shape(self, input_shape):
        return [input_shape[0], self.num_hidden]

inputs = Input(shape = (5, ))
x = CustomLayer(32)(inputs)
model = Model(inputs, x)

