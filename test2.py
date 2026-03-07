import os
import numpy as np
import librosa
import kagglehub
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- 1. KONFIGURACJA ---
print("Downloading a dataset...")
path = kagglehub.dataset_download("jbuchner/synthetic-speech-commands-dataset")

DATASET_PATH = path 
commands = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
label_string = tf.strings.split(DATASET_PATH, os.path.sep)[-2]
label_id = tf.argmax(label_string == commands)
