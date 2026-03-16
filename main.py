import os
import cv2
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model
from text_processing import text_processing
from tokens import tokens_feeding
from model_functions import generate_caption
from music_data_processing import extract_spectrogram

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mapping = text_processing(r'.\captions.txt')
tokenizer = tokens_feeding(mapping)

audio_model = load_model("audio_model_v1.h5")
caption_model = load_model("model_final_epoch_20.keras")

DIGITS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
IMG_HEIGHT = 64

current_command = None
is_running = True

def listen_and_predict():
    global current_command, is_running
    while is_running:
        audio = sd.rec(int(16000), samplerate=16000, channels=1, dtype='float32')
        sd.wait()
        
        if np.max(np.abs(audio)) < 0.05:
            continue
            
        wavfile.write("temp_audio.wav", 16000, audio)
        
        try:
            spec = extract_spectrogram("temp_audio.wav", IMG_HEIGHT)
            if spec is not None:
                spec = np.expand_dims(spec, axis=0)
                pred = audio_model.predict(spec, verbose=0)
                best_pred = np.max(pred)
                predicted_digit = DIGITS[np.argmax(pred)]
                
                if best_pred > 0.75:
                    print(f"\n[VOICE COMMAND]: {predicted_digit} (Confidence: {best_pred*100:.0f}%)")
                    current_command = predicted_digit
        except Exception:
            pass

print("\n" + "="*45)
print("       VISION & VOICE CONTROL SYSTEM")
print("="*45)
print(" INSTRUCTIONS - AVAILABLE VOICE COMMANDS:")
print("  'one'   -> Enable camera preview")
print("  'two'   -> Capture photo and generate caption")
print("  'three' -> Display README.md content")
print("  'four'  -> Exit the application")
print("="*45 + "\n")

listener_thread = threading.Thread(target=listen_and_predict, daemon=True)
listener_thread.start()

cap = None
camera_enabled = False

try:
    while is_running:
        if current_command == "one":
            if not camera_enabled:
                print("Camera ENABLED")
                cap = cv2.VideoCapture(0)
                camera_enabled = True
            current_command = None

        elif current_command == "two":
            if camera_enabled and cap is not None:
                print("Capturing photo and analyzing...")
                ret, frame = cap.read()
                if ret:
                    cv2.imwrite("temp_frame.jpg", frame)
                    result = generate_caption(caption_model, tokenizer, "temp_frame.jpg", 38)
                    print(f"I see: {result}")
            else:
                print("Error: Enable camera first (say 'one')!")
            current_command = None

        elif current_command == "three":
            print("\n--- README CONTENT ---")
            if os.path.exists("README.md"):
                with open("README.md", "r", encoding="utf-8") as f:
                    print(f.read())
            else:
                print("Error: README.md file not found.")
            print("----------------------\n")
            current_command = None

        elif current_command == "four":
            print("Shutting down system...")
            is_running = False
            current_command = None

        if camera_enabled and cap is not None:
            ret, frame = cap.read()
            if ret:
                cv2.imshow("Camera Stream", frame)
        
        if cv2.waitKey(10) & 0xFF == 27:
            print("ESC pressed. Exiting...")
            is_running = False
            break

except KeyboardInterrupt:
    print("\nApplication interrupted...")
    is_running = False

finally:
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")
    if os.path.exists("temp_frame.jpg"):
        os.remove("temp_frame.jpg")