import os
import numpy as np
import librosa
import kagglehub
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt # <-- DODANA BIBLIOTEKA DO RYSOWANIA
from music_data_processing import data_processing, extract_spectrogram
from music_neural_network import msn

path = kagglehub.dataset_download("jbuchner/synthetic-speech-commands-dataset")

DATASET_PATH = path 
DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
SAMPLES_PER_DIGIT = 1000  
IMG_HEIGHT = 64

X_train, X_test, y_train, y_test, input_shape, DIGITS = data_processing(DATASET_PATH, DIGITS, SAMPLES_PER_DIGIT, IMG_HEIGHT)

model = msn(input_shape, DIGITS)

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# --- 4. TRENOWANIE ---
print(f"\nRozpoczynam trenowanie na {len(X_train)} próbkach...")
model.fit(X_train, y_train, epochs=15, batch_size=32, 
          validation_data=(X_test, y_test))

# --- 5. FINALNY WYNIK ---
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✅ FINALNE ACCURACY: {acc*100:.2f}%")

model.save("audio_model_v1.h5")
print("Model zapisany.")

