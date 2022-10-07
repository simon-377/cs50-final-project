import gc
import os
import random
import tensorflow as tf
import tensorflow_io as tfio
import csv
import math
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, Conv2DTranspose


# A lot of code is somewhat based on "Nicholas Renotte" tutorial on YouTube, video: "Build a Deep Audio Classifier with Python and Tensorflow"

BAD_FILES = ['UrbanSound8K/audio/fold1/101415-3-0-2.wav', 'UrbanSound8K/audio/fold1/160094-3-0-0.wav',
    'UrbanSound8K/audio/fold1/87275-1-0-0.wav', 'UrbanSound8K/audio/fold1/87275-1-1-0.wav', 
    'UrbanSound8K/audio/fold1/87275-1-2-0.wav', 'UrbanSound8K/audio/fold1/87275-1-3-0.wav', 
    'UrbanSound8K/audio/fold1/87275-1-4-0.wav', 'UrbanSound8K/audio/fold1/87275-1-5-0.wav', 'UrbanSound8K/audio/fold2/147672-3-0-0.wav', 
    'UrbanSound8K/audio/fold2/147672-3-1-0.wav', 'UrbanSound8K/audio/fold2/147672-3-2-0.wav', 'UrbanSound8K/audio/fold2/155129-1-0-0.wav', 
    'UrbanSound8K/audio/fold2/155129-1-1-0.wav', 'UrbanSound8K/audio/fold2/160092-3-0-0.wav', 'UrbanSound8K/audio/fold2/17307-1-0-0.wav', 
    'UrbanSound8K/audio/fold2/4201-3-0-0.wav', 'UrbanSound8K/audio/fold2/4911-3-0-0.wav', 'UrbanSound8K/audio/fold2/76091-6-2-0.wav', 
    'UrbanSound8K/audio/fold2/76091-6-3-0.wav', 'UrbanSound8K/audio/fold2/76091-6-4-0.wav', 'UrbanSound8K/audio/fold3/118070-1-0-0.wav', 
    'UrbanSound8K/audio/fold3/118496-1-0-0.wav', 'UrbanSound8K/audio/fold3/118496-1-1-0.wav', 'UrbanSound8K/audio/fold3/151359-1-0-0.wav', 
    'UrbanSound8K/audio/fold3/151359-1-1-0.wav', 'UrbanSound8K/audio/fold3/151359-1-2-0.wav', 'UrbanSound8K/audio/fold3/151359-1-3-0.wav', 
    'UrbanSound8K/audio/fold3/160093-3-0-0.wav', 'UrbanSound8K/audio/fold3/18594-1-2-0.wav', 'UrbanSound8K/audio/fold3/18594-1-3-0.wav', 
    'UrbanSound8K/audio/fold3/18594-1-4-0.wav', 'UrbanSound8K/audio/fold4/47926-3-1-0.wav', 'UrbanSound8K/audio/fold5/26177-1-0-0.wav', 
    'UrbanSound8K/audio/fold5/43803-1-0-0.wav', 'UrbanSound8K/audio/fold5/71439-1-1-0.wav', 'UrbanSound8K/audio/fold6/162702-1-0-0.wav', 
    'UrbanSound8K/audio/fold6/43802-1-0-0.wav', 'UrbanSound8K/audio/fold6/43802-1-1-0.wav', 'UrbanSound8K/audio/fold6/4912-3-0-0.wav', 
    'UrbanSound8K/audio/fold7/170243-1-0-0.wav', 'UrbanSound8K/audio/fold7/43784-3-0-0.wav', 'UrbanSound8K/audio/fold8/155313-3-0-0.wav', 
    'UrbanSound8K/audio/fold9/155130-1-0-0.wav', 'UrbanSound8K/audio/fold9/180156-1-12-0.wav', 'UrbanSound8K/audio/fold9/27068-1-0-0.wav', 
    'UrbanSound8K/audio/fold10/17124-1-0-0.wav', 'UrbanSound8K/audio/fold10/26255-3-0-0.wav']

LENGTH = .5
RATE_OUT = 22050
LENGTH_INT = tf.cast(LENGTH * RATE_OUT, dtype=tf.int32)

def main():

    # Get metadata and cleanup
    metadata = meta()
    # metadata[0] is paths, metadata[1] is labels. After cleanup. "Letters" no longer matters but is needed of .map.
    metadata = data_cleanup(metadata)

    letters = tf.data.Dataset.from_tensor_slices(metadata[1])
    paths = tf.data.Dataset.from_tensor_slices(metadata[0])
    data = tf.data.Dataset.zip((paths, letters))

    # Small batch size helped with RAM issues
    data = data.shuffle(buffer_size=12000)
    data = data.take(math.floor(len(data) * .3))
    data = data.map(load_wav)
    data = data.cache()
    data = data.batch(1)
    data = data.prefetch(0)
    
    # Transpose convolutions invert the modifications to the size of the matrix
    # This allows us to get same output size as input size
    # I assume convolutions are a good idea... not sure though
    model = Sequential()
    model.add(Conv2DTranspose(16, (3,3), activation='relu', input_shape=(329, 257, 2)))
    model.add(Conv2DTranspose(16, (3,3), activation='relu'))
    model.add(Conv2DTranspose(16, (3,3), activation='relu'))
    model.add(Conv2DTranspose(16, (3,3), activation='relu'))
    model.add(Conv2DTranspose(16, (3,3), activation='relu'))
    model.add(Conv2DTranspose(16, (3,3), activation='relu'))
    model.add(Conv2DTranspose(16, (3,3), activation='relu'))
    model.add(Conv2DTranspose(16, (3,3), activation='relu'))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Conv2D(2,  (3,3), activation='relu'))

    checkpoint_path = "cp_noise"
    #model.load_weights(checkpoint_path)
    
    # I tried "mae" and "mse" loss functions and both seem fine
    # I never experimented with the optimizer
    model.compile(optimizer='Adam', loss='mean_absolute_error', metrics=[])
    print(model.summary())
    
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=False)

    hist = model.fit(data, epochs=1, callbacks=[cp_callback, MyCustomCallback()])
    model.save('model')    


# Create list from csv rows: (filename | bad text | good text)
def meta():
    with open("LJSpeech-1.1/metadata.csv", 'r') as metadata:
        line = csv.reader(metadata, delimiter="|",)
        return [row for row in line]

    
# We no longer need to do a bunch of cleanup, we just load all the file paths and put a placeholder for letters.
def data_cleanup(metadata):
    letters = []
    paths = []
    for row in metadata:
        row[0] = os.path.join("LJSpeech-1.1", "wavs", (row[0] + ".wav"))
        try:
            letters.append(1)
            paths.append(row[0])
        except:
            continue
    return [paths, letters]


# You don't just get to write all python code in here, see tf documentation (.map)
def load_wav(filename, label):
    # Load encoded .wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, samplerate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    samplerate = tf.cast(samplerate, dtype="int64")
    # Change audio from 44100hz to 16000hz
    wav = tfio.audio.resample(wav, rate_in=samplerate, rate_out=RATE_OUT)

    wav_random_offset = tf.py_function(func=random_chunck, inp=[wav], Tout=tf.int32)    
    wav = wav[wav_random_offset:wav_random_offset + LENGTH_INT]

    bg_wav = load_background()
    bg_wav_random_offset = tf.py_function(func=random_chunck, inp=[bg_wav], Tout=tf.int32)
    bg_wav = bg_wav[bg_wav_random_offset:bg_wav_random_offset + LENGTH_INT]

    # Pick mix
    rand = tf.py_function(func=random_number, inp=[3], Tout=tf.int32)
    rand = tf.cast(rand, tf.float32)

    # Mix speech with around 30% background
    label = wav
    bg_wav = bg_wav * ((4 + rand) / 10)
    wav = wav + bg_wav
    normalize_ = tf.reduce_max(tf.abs(wav))
    wav = wav / normalize_

    # Make spectrogram. In the last axis, there is usually 3 rgb values.
    # We split our spectrogram into complex and real and put those two in there.
    spectrogram = tf.signal.stft(wav, frame_length=512, frame_step=32)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.concat([tf.math.real(spectrogram), tf.math.imag(spectrogram)], axis=2)

    # Same thing for the label
    label = tf.signal.stft(label, frame_length=512, frame_step=32)
    label = tf.expand_dims(label, axis=2)
    label = tf.concat([tf.math.real(label), tf.math.imag(label)], axis=2)

    return spectrogram, label

def load_background():
    random_file = tf.py_function(func=get_random_bg_path, inp=[], Tout=tf.string)
    
    # Should have probably made this its own function instead of copying it
    # Load encoded .wav file
    bg_file_contents = tf.io.read_file(random_file)
    # Decode wav (tensors by channels)
    bg_wav, bg_samplerate = tf.audio.decode_wav(bg_file_contents, desired_channels=1)
    print(bg_wav)
    print()
    # Removes trailing axis
    bg_wav = tf.squeeze(bg_wav, axis=-1)
    bg_samplerate = tf.cast(bg_samplerate, dtype="int64")
    # Change audio from 22050
    bg_wav = tfio.audio.resample(bg_wav, rate_in=bg_samplerate, rate_out=RATE_OUT)

    print(bg_wav.shape)
    
    bg_zero_padding = tf.zeros(tf.cast(RATE_OUT * LENGTH, dtype=tf.int32), dtype=tf.float32)
    bg_wav = tf.concat([bg_wav, bg_zero_padding], 0)
    return bg_wav


def get_random_bg_path():
    # Get folder paths
    base_folder = os.path.join("UrbanSound8K", "audio")
    folders = [os.path.join(base_folder, "fold") + str(i) for i in range(1,11)]

    # Pick random folder
    random_folder = random.choice(folders)
    random_file = os.path.join(random_folder, random.choice(os.listdir(random_folder)))
    if random_file in BAD_FILES:
        return get_random_bg_path()
    return random_file


def random_chunck(shape):
    return random.randrange(int(shape.shape - LENGTH_INT - 1))


# Hopefully fixes memory leak https://github.com/tensorflow/tensorflow/issues/31312 ; Doesn't seem to matter much
class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print()
        print("clear")
        print()

        gc.collect()
        tf.keras.backend.clear_session()


# Getting "random" to work required using py_function since .map broke it otherwise
def random_number(a):
    return random.randrange(a)


# Pass a string to this from load_wav() and load_background() to troubleshoot
def print_me(string_):
    print()
    print(str(string_))
    print()
    return 1


main()