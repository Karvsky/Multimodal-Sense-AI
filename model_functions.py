import os
import pickle
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, Add
from tensorflow.keras.models import Model
from image_processing import data_import

def get_all_features(directory):
    if os.path.exists('features.pkl'):
        return pickle.load(open('features.pkl', 'rb'))
    
    base_model = InceptionV3(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    features = {}
    img_array, img_label = data_import(directory)
    
    for image, img_name in zip(img_array, img_label):
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image.astype(np.float32))
        feature = model.predict(image, verbose=0)
        features[img_name] = feature.reshape(-1)
    
    pickle.dump(features, open('features.pkl', 'wb'), protocol=4)
    return features

def data_generator(descriptions, photos, tokenizer, max_length, vocab_size, batch_size):
    X1, X2, y = list(), list(), list()
    n = 0
    keys = list(descriptions.keys()) 
    
    while True:
        random.shuffle(keys) 
        
        for key in keys:
            if key not in photos: continue
            n += 1
            photo = photos[key]
            for desc in descriptions[key]:
                for i in range(1, len(desc)):
                    in_seq, out_seq = desc[:i], desc[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            if n == batch_size:
                yield (np.array(X1), np.array(X2)), np.array(y)
                X1, X2, y = list(), list(), list()
                n = 0

def build_model(vocab_size, max_length=38):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    
    decoder1 = Add()([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

def generate_caption(model, tokenizer, photo_path, max_length=38):
    base_model = InceptionV3(weights='imagenet')
    feat_model = Model(inputs=base_model.input, outputs=base_model.layers[-2].output)
    
    img = cv2.imread(photo_path)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (299, 299))
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img.astype(np.float32))
        photo_feature = feat_model.predict(img, verbose=0)
    else:
        return "Błąd ładowania obrazu"

    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat)
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word
    return in_text