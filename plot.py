# %%
import tensorflow as tf
import tensorflow_io as tfio
import soundfile
from matplotlib import pyplot as plt

TEST_BG = 'UrbanSound8K/audio/fold1/22962-4-0-0.wav'
TEST_FG = 'LJSpeech-1.1/wavs/LJ002-0013.wav'

TEST = 'test.wav'
OUTPUT = 'output.wav'
HIGH_PASS = 'high-pass.wav'

RATE_OUT = 22050
WAVES = [TEST_BG, TEST_FG, TEST, OUTPUT]

def main():
    # Load wav
    content = tf.io.read_file(HIGH_PASS)
    wav, samplerate = tf.audio.decode_wav(content, desired_channels=2)
    # Mix both channels
    wav = wav[:,0] + wav[:,1] / 2
    samplerate = tf.cast(samplerate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, samplerate, rate_out=RATE_OUT)
    #wav = wav[RATE_OUT * 7:RATE_OUT * 10]

    # Make plot
    plt.figure(figsize=(64,21))
    plt.plot(tf.transpose(wav))
    plt.show()


main()
# %%
