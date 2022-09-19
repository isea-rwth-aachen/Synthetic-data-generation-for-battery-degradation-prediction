# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 19:03:15 2022

@author: HarshvardhanSamsukha
"""

from sklearn.gaussian_process.kernels import Matern
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, Dense, MaxPooling1D, Dropout, Flatten
)
from tensorflow.keras.optimizers import Adam


def cnn_model(complete_inputs, targets, val_inputs, val_targets, callback):
    filters = 256
    model_cnn = Sequential()
    model_cnn.add(Conv1D(
        filters=filters, kernel_size=3, activation='relu', 
        input_shape=(complete_inputs.shape[1],complete_inputs.shape[2])
    ))
    model_cnn.add(MaxPooling1D(pool_size=2))
    model_cnn.add(Dropout(0.1))
    model_cnn.add(
        Conv1D(filters=filters/2, kernel_size=3, activation='relu')
    )
    model_cnn.add(
        Conv1D(filters=filters/2, kernel_size=3, activation='relu')
    )
    model_cnn.add(Flatten())
    model_cnn.add(Dense(20))
    model_cnn.add(Dropout(0.1))
    model_cnn.add(Dense(1))
    model_cnn.compile(optimizer=Adam(0.0001), loss = 'mae')
    model_cnn.fit(
        complete_inputs, targets, batch_size=5, epochs=700, 
        shuffle=True, validation_data=(val_inputs, val_targets), 
        callbacks=[callback], verbose=False
    )
    return model_cnn

def gpr_model(X_train, y_train):
    kernel=Matern(length_scale=1, length_scale_bounds=(20,200), nu=1.5)
    # Gaussian process regressor
    gp = gpr(kernel=kernel, n_restarts_optimizer=3000, alpha=5e-2)
    gp.fit(X_train, y_train)
    return gp