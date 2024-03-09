import tensorflow as tf
import librosa
import os
import numpy as np

def load_wav_16k_mono(filename):
    if tf.is_tensor(filename):
        filename = filename.numpy().decode('utf-8')
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    #wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    
    wav, sr = librosa.load(filename, sr=None)
    wav_resampled = librosa.resample(wav, orig_sr=sr, target_sr=16000)

    return wav_resampled
# Function to load and preprocess the audio file
def preprocess(file_path, label):
    if tf.is_tensor(file_path):
        file_path_str = file_path.numpy().decode('utf-8')
    else:
        file_path_str = file_path  
    wav = load_wav_16k_mono(file_path_str)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram, label

# Load the saved model
BASE_PATH = '/Users/harsh/Projects/coocket_whistle_python/Data'
model = tf.keras.models.load_model(os.path.join(BASE_PATH, 'testmodel'))

# Function to make a prediction on an audio file
def predict(file_path):
    # Preprocess the audio file
    test_spectrogram, _ = preprocess(file_path, 0)  # Assuming label doesn't matter for testing

# Ensure the input is in the correct shape for the model
    test_spectrogram = tf.expand_dims(test_spectrogram, 0)

    # Predict and return the result
    prediction = model.predict(test_spectrogram)
    return prediction

# Example usage
test_file_path = os.path.join(BASE_PATH, 'Split Training Data WAV/output/Only_whistle_1.wav') 
prediction = predict(test_file_path)
print("Prediction:", prediction)
