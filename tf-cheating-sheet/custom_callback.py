#!/usr/bin/env python
# coding: utf-8


# -----------------------------------------
# Learning Rate Scheduling callback
# -----------------------------------------

from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

# Callback 클래스 상속
class CustomLearningLateCallback(Callback):
    def __init__(self):
        pass
    
    def on_train_begin(self, logs = None):
        pass
    
    def on_train_end(self, logs = None):
        pass
    
    def on_train_batch_begin(self, batch, logs = None):
        pass
    
    def on_train_batch_end(self, batch, logs = None):
        pass
    
    # 0.1배 감소 연산
    def down_lr(self, current_lr):
        return current_lr * 0.1
    
    def on_epoch_begin(self, epoch, logs = None):
        current_lr = self.model.optimizer.lr
        
        if(epoch > 1):
            # 5, 8, 10번째마다 학습률을 감소
            if((epoch == 4) or (epoch == 7) or (epoch == 9)):
                current_lr = self.down_lr(current_lr)
                
                # 감소된 학습률을 현재 모델 옵티마이저의 학습률로 설정
                K.set_value(self.model.optimizer.lr, current_lr)
                print('\nEpoch %03d: learning rate change! %s.' % (epoch + 1, current_lr.numpy()))
                
    def on_epoch_end(self, epoch, logs = None):
        pass


model.fit(x_train, y_train,
         batch_size = 32,
         validation_data = (x_val, y_val),
         epochs = 10,
         callbacks = [CustomLearningLateCallback()])


# -----------------------------------------
# CosineAnnealing Learning Rate
# -----------------------------------------
class CosineAnnealingLearningRateSchedule(Callback):
    def __init__(self, n_epochs, init_lr, T_mult = 1, eta_min = 0,restart_decay = 0, verbose = 0):
        self.T_max = n_epochs
        self.T_mult = T_mult
        self.cycle_cnt = 0
        self.restart_decay = restart_decay
        self.init_lr = init_lr
        self.eta_min = eta_min
        self.lrates = list()
    
    # caculate learning rate for an epoch
    def cosine_annealing(self, epoch):
        lr = self.eta_min + (self.init_lr - self.eta_min) * (1 + math.cos(math.pi * (epoch / self.T_max))) / 2
        if(epoch == self.T_max):
            self.cycle_cnt += 1
            self.T_max = self.T_mult * self.T_max

        if(self.restart_decay >0):
            self.init_lr *= self.restart_decay
            print('change init learning rate {}'.format(self.init_lr))
    return lr
    
    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs = None):
        lr = self.cosine_annealing(epoch)
        print('\nEpoch %05d: CosineAnnealingScheduler setting learng rate to %s.' % (epoch + 1, lr))
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)

