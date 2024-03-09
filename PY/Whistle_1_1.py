# %%
#!pip install tensorflow==2.8.0 tensorflow-gpu==2.8.0 tensorflow-io matplotlib
!pip install tensorflow matplotlib
!pip install tensorflow-gpu
import tensorflow as tf

!pip install tensorflow_io
!pip install librosa


# %%
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
import librosa
import numpy as np


# %%
# Example: Update with the actual path to your 'Split Training Data' folder
os.listdir('/Users/harsh/Projects/coocket_whistle_python/Data/Not Whistles')

BASE_PATH = '/Users/harsh/Projects/coocket_whistle_python/Data'

# Now, you can join this base path with your individual file names
CAPUCHIN_FILE = os.path.join(BASE_PATH, 'Split Training Data WAV/output/Only_whistle_1.wav')  # Example file name
NOT_CAPUCHIN_FILE = os.path.join(BASE_PATH, 'Not Whistles/afternoon-birds-song-in-forest-0.wav')  # Example file name

# %%
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

# %%
wave = load_wav_16k_mono(CAPUCHIN_FILE)
nwave = load_wav_16k_mono(NOT_CAPUCHIN_FILE)

# %%
plt.plot(wave)
plt.plot(nwave)
plt.show()

# %%
POS = os.path.join(BASE_PATH, 'Split Training Data WAV/output')
NEG = os.path.join(BASE_PATH, 'Not Whistles')

# %%
pos = tf.data.Dataset.list_files(POS+'/*.wav')
neg = tf.data.Dataset.list_files(NEG+'/*.wav')

# %%
def get_dataset_size(dataset):
    count = 0
    for _ in dataset:
        count += 1
    return count

pos_size = get_dataset_size(pos)
neg_size = get_dataset_size(neg)

positives = tf.data.Dataset.zip((pos, tf.data.Dataset.from_tensor_slices(tf.ones(pos_size))))
negatives = tf.data.Dataset.zip((neg, tf.data.Dataset.from_tensor_slices(tf.zeros(neg_size))))
data = positives.concatenate(negatives)


# %%
neg_size

# %%
lengths = []
for file in os.listdir(os.path.join(BASE_PATH, 'Split Training Data WAV/output')):
    if file.endswith('.wav'):
        #wav, sr = librosa.load(os.path.join(BASE_PATH, 'Split Training Data WAV', file), sr=16000, mono=True)
        #lengths.append(len(wav))
        tensor_wave = load_wav_16k_mono(os.path.join(BASE_PATH, 'Split Training Data WAV/output', file))
        lengths.append(len(tensor_wave))

# %%
tf.math.reduce_mean(lengths)

# %%
tf.math.reduce_mean(lengths)

# %%
tf.math.reduce_max(lengths)

# %%
def preprocess(file_path, label):
    # [wav, label] = tf.py_function(load_wav_16k_mono, [file_path], [tf.float32, label.dtype])
    # wav.set_shape([48000])
    # zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    # wav = tf.concat([zero_padding, wav], 0)
    # spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    # spectrogram = tf.abs(spectrogram)
    # spectrogram = tf.expand_dims(spectrogram, -1)
    # return spectrogram, label
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

# %%
filepath, label = positives.shuffle(buffer_size=10000).as_numpy_iterator().next()

# %%
spectrogram, label = preprocess(filepath, label)

# %%
plt.figure(figsize=(30,20))
plt.imshow(tf.transpose(spectrogram)[0])
plt.show()

# %%
def tf_preprocess(file_path, label):
    [spectrogram, label] = tf.py_function(preprocess, [file_path, label], [tf.float32, label.dtype])
    return spectrogram, label

data = data.map(tf_preprocess)
data = data.cache()
data = data.shuffle(buffer_size=1000)
data = data.batch(19)
data = data.prefetch(8)

# %%
train = data.take(5)
test = data.skip(5).take(1)

# %%
for samples, labels in train.take(3):
    display("aaa")


# %%
samples, labels = train.as_numpy_iterator().next()

# %%
samples.shape

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten

# %%
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# %%
model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])

# %%
model.summary()

# %%
hist = model.fit(train, epochs=4, validation_data=test)

# %%
plt.title('Loss')
plt.plot(hist.history['loss'], 'r')
plt.plot(hist.history['val_loss'], 'b')
plt.show()

# %%
hist.history

# %%
plt.title('Precision')
plt.plot(hist.history['precision_1'], 'r')
plt.plot(hist.history['val_precision_1'], 'b')
plt.show()

# %%
plt.title('Recall')
plt.plot(hist.history['recall_1'], 'r')
plt.plot(hist.history['val_recall_1'], 'b')
plt.show()

# %%
os.listdir('/Users/harsh/Projects/coocket_whistle_python/Data/Not Whistles')

BASE_PATH = '/Users/harsh/Projects/coocket_whistle_python/Data'

# Now, you can join this base path with your individual file names
#test_file_path = os.path.join(BASE_PATH, 'Split Training Data WAV/output/Only_whistle_set_3 (46).wav') 
test_file_path = os.path.join(BASE_PATH, 'Not Whistles/afternoon-birds-song-in-forest-0.wav') 

# Step 1: Load the test audio file
test_audio = load_wav_16k_mono(test_file_path)

# Step 2: Preprocess the test audio
test_spectrogram, _ = preprocess(test_file_path, 0)  # Assuming label doesn't matter for testing

# Ensure the input is in the correct shape for the model
test_spectrogram = tf.expand_dims(test_spectrogram, 0)

# Step 3: Predict
prediction = model.predict(test_spectrogram)
print("Prediction:", prediction)
confidence = prediction[0][0]
print("Conf:", confidence)



# %%
#model.save( os.path.join(BASE_PATH, 'testmodel'))


