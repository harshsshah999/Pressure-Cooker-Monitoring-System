from contextlib import redirect_stdout
from io import StringIO
from flask import Flask, request, jsonify
import tensorflow as tf
import librosa
import os
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# PythonRunner class from Code1
class PythonRunner:
    __globals = {}
    __locals = {}

    def run(self, code):
        f = StringIO()
        with redirect_stdout(f):
            exec(code, self.__globals, self.__locals)
        return f.getvalue()

pr = PythonRunner()

# Load the TensorFlow model from Code2
BASE_PATH = os.environ.get('AUDIO_DATA_PATH', '/path/to/your/model/directory')
model = tf.keras.models.load_model(os.path.join(BASE_PATH, 'testmodel'))

# Functions from Code2
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

def predict(file_path):
    # Preprocess the audio file
    test_spectrogram, _ = preprocess(file_path, 0)  # Assuming label doesn't matter for testing

# Ensure the input is in the correct shape for the model
    test_spectrogram = tf.expand_dims(test_spectrogram, 0)

    # Predict and return the result
    prediction = model.predict(test_spectrogram)
    return prediction

# Flask routes
@app.route("/")
def hello_world():
    return 'Enter Python code and tap "Run".'

@app.route("/python", methods=["POST"])
def run_python():
    try:
        return pr.run(request.json["command"])
    except Exception as e:
        return str(e)

@app.route("/predict", methods=["POST"])
def make_prediction():
    try:
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        # You might need to save the file temporarily if librosa requires a file path
        file_path = "/path/to/temporary/storage/" + file.filename
        file.save(file_path)
        prediction = predict(file_path)
        os.remove(file_path)  # Delete the temporary file after prediction
        confidence = prediction[0][0]
        return jsonify({"prediction": confidence})
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    port = 55001
    print("Trying to run a socket server on:", port)
    app.run(port=port)
