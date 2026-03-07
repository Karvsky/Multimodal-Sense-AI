import os
import numpy as np
import librosa
import kagglehub
import tensorflow as tf
import keras
from keras import layers, models



def msn(input_shape, DIGITS):

    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.BatchNormalization(),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(len(DIGITS), activation='softmax')
    ])

    return model