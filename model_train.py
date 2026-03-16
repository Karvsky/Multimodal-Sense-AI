import os
import tensorflow as tf
from text_processing import text_processing
from tokens import tokens_feeding, tokens
from model_functions import get_all_features, data_generator, build_model, generate_caption

PATH_CAPTIONS = r'.\captions.txt'
PATH_IMAGES = r'.\Images' 
TEST_IMAGE = r'.\Images\1000268201_693b08cb0e.jpg'

mapping = text_processing(PATH_CAPTIONS)
tokenizer = tokens_feeding(mapping)
mapping_sequences = tokens(mapping) 
vocab_size = len(tokenizer.word_index) + 1

all_features = get_all_features(PATH_IMAGES)

model = build_model(vocab_size, 38)

epochs = 20 
batch_size = 32
steps = len(mapping_sequences) // batch_size

output_signature = (
    (
        tf.TensorSpec(shape=(None, 2048), dtype=tf.float32), 
        tf.TensorSpec(shape=(None, 38), dtype=tf.float32)    
    ),
    tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32) 
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(mapping_sequences, all_features, tokenizer, 38, vocab_size, batch_size),
    output_signature=output_signature
)

os.makedirs('modele_epoki', exist_ok=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='modele_epoki/model_epoch_{epoch:02d}.keras', 
    save_freq='epoch',                                     
    save_best_only=False, 
    verbose=1                           
)

model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps, verbose=1, callbacks=[checkpoint])

model.save(f'model_final_epoch_{epochs}.keras') 

if os.path.exists(TEST_IMAGE):
    result = generate_caption(model, tokenizer, TEST_IMAGE, 38)
    print(result)