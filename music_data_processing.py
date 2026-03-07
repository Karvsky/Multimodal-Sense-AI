import os
import numpy as np
import librosa
import kagglehub
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder




def extract_spectrogram(file_path, IMG_HEIGHT):
    try:

        audio, sr = librosa.load(file_path, duration=1.0, sr=16000)

        if len(audio) < 16000:
            audio = np.pad(audio, (0, 16000 - len(audio)))
            
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=IMG_HEIGHT)

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-6)
        return log_mel_spec[..., np.newaxis] 
    except Exception as e:
        return None

def data_processing(DATASET_PATH, DIGITS, SAMPLES_PER_DIGIT, IMG_HEIGHT):

    X, y = [], []


    actual_data_path = DATASET_PATH
    for root, dirs, files in os.walk(DATASET_PATH):
        if all(d in dirs for d in ['zero', 'one', 'two']):
            actual_data_path = root
            break

    for digit in DIGITS:
        folder_path = os.path.join(actual_data_path, digit)

        all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.wav')]
        selected_files = all_files[:SAMPLES_PER_DIGIT]
        
        for f in selected_files:
            spec = extract_spectrogram(os.path.join(folder_path, f), IMG_HEIGHT)
            if spec is not None:
                X.append(spec)
                y.append(digit)

    if len(X) == 0:
        print("❌ ERROR")
        exit()

    X = np.array(X)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    input_shape = X.shape[1:]
    print(f"The network's output size is: {input_shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, input_shape, DIGITS