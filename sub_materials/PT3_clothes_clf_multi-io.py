#!/usr/bin/env python
# coding: utf-8

# > 예제가 좀 구림. multi IO 인데<br/>
# (image pixel values, cate-colors) -> (one-hot colors, one-hot types)

# In[1]:


import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Concatenate
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


# In[2]:


DATA_PATH = './csv_data/colorinfo'

train_df = pd.read_csv(DATA_PATH + '/train_color.csv')
val_df = pd.read_csv(DATA_PATH + '/val_color.csv')
test_df = pd.read_csv(DATA_PATH + '/test_color.csv')


# In[3]:


# train_df.head()


# # ** define custom generator

# In[4]:


def get_steps(num_samples, batch_size):
    if (num_samples % batch_size) > 0:
        return (num_samples // batch_size) + 1
    else:
        return num_samples // batch_size


# In[8]:


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size = 32, 
                 target_size = (122, 122), shuffle = True):
        self.len_df = len(df)
        self.batch_size = batch_size
        self.target_size = target_size
        self.shuffle = shuffle
        self.class_col = ['black', 'blue', 'brown', 'green', 'red', 'white', 
             'dress', 'shirt', 'pants', 'shorts', 'shoes']
        
        # 이미지 불러오기
        self.generator = ImageDataGenerator(rescale = 1./255)
        self.df_generator =            self.generator.flow_from_dataframe(dataframe = df,
                                               directory = '',
                                               x_col = 'image',
                                               y_col = self.class_col,
                                               target_size = self.target_size,
                                               color_mode='rgb',
                                               class_mode='raw',
                                               batch_size=self.batch_size,
                                               shuffle = True,
                                               seed=42)
        self.colors_df = df['color']
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.floor(self.len_df) / self.batch_size)
    
    def on_epoch_end(self):
        self.indexes = np.arange(self.len_df)
        if self.shuffle:
            np.random.shuffle(self.indexes)
            
    def __data_generation(self, indexes):
        colors = np.array([self.colors_df[k] for k in indexes])
        return colors
    
    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        colors = self.__data_generation(indexes)
        images, labels = self.df_generator.__getitem__(index)
        
        # return multi-input and output
        return [images, colors], labels


# In[9]:


train_datagen = DataGenerator(train_df)
val_datagen = DataGenerator(val_df)


# # ** build model

# In[10]:


def get_model():
    # 2개의 input층 정의 
    img_input = Input(shape = (112, 112, 3))
    color_input = Input(shape = [1])
    
    # raw image로부터 feature extracting
    x = Conv2D(32, (3, 3), padding = "same", activation = "relu")(img_input)
    x = MaxPooling2D((3, 3), strides = 2)(x)
    x = Conv2D(64, (3, 3), padding = "same", activation = "relu")(x)
    x = MaxPooling2D((3, 3), strides = 2)(x)
    x = Conv2D(64, (3, 3), padding = "same", activation = "relu")(x)
    x = MaxPooling2D((3, 3), strides = 2)(x)
    x = GlobalAveragePooling2D()(x)
    
    # categorical color variable을 하나의 설명변수로써 concat
    color_concat = Concatenate()([x, color_input])
    
    x = Dense(64, activation = "relu")(color_concat)
    x = Dense(11, activation = "sigmoid")(x)
    
    model = Model(inputs = [img_input, color_input], outputs = x)
    model.compile(optimizer = "adam",
                  loss = "binary_crossentropy",
                  metrics = ['acc'])
    
    return model

model = get_model()


# # ** train

# In[ ]:


batch_size = 32

history = model.fit(train_datagen,
         validation_data = val_datagen,
         epochs = 10)


# In[ ]:


plt.plot(history.hitory)

