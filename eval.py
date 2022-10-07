# %%
import tensorflow as tf
import tensorflow_io as tfio
import soundfile


# LENGTH must be smaller or equal to 1
# If you change it here, you must change it in noise.py as well
# It will complain, change "input_size=" first field to "actual =" size at the bottom of the bug message
# It sais "Conv2DCustomBackdropInput" at the beginning of the line ^
# Wasn't sure how to automate that
# Don't change RATE_OUT
LENGTH = .5
RATE_OUT = 22050
LENGTH_INT = tf.cast(LENGTH * RATE_OUT, dtype=tf.int32)
# Change this file path to your .wav, that's all
WAV = 'test.wav'

TEST_BG = 'UrbanSound8K/audio/fold1/22962-4-0-0.wav'
TEST_FG = 'LJSpeech-1.1/wavs/LJ002-0013.wav'


def main():
    #make_test(get_file(TEST_BG), get_file(TEST_FG))

    wav = get_file(WAV)

    model = tf.keras.models.load_model('model')
    print(model.summary())

    print('Removing noise from input file')
    step = LENGTH_INT
    for i in range(0, wav.shape[0], step):
        # Make spectrogram in "LENGTH" steps and concatenate real and imaginary part into last dimension
        spectrogram = make_spectrogram(wav[i:i + (LENGTH_INT)])
        spectrogram = tf.concat([tf.math.real(spectrogram), tf.math.imag(spectrogram)], axis=3)
        try:
            tmp = model(spectrogram, training=False)
            # If bad input size, take the last "LENGTH" segment and concatenate only the remainder
            # Length of the .wav is not usually divisible by LENGTH, which causes us to step outside of the list indexes
        except ValueError:
            shape_tmp = spectrogram.shape
            spectrogram = make_spectrogram(wav[-(LENGTH_INT + 1):-1])
            spectrogram = tf.concat([tf.math.real(spectrogram), tf.math.imag(spectrogram)], axis=3)
            tmp = model(spectrogram, training=False)
            tmp = tmp[:,-shape_tmp[1] - 1:-1,:,:]

        output_spec = tmp if i == 0 else tf.concat([output_spec, tmp], axis=1)
    
    # 2D Real -> 1D Complex in last dimension
    output_spec = tf.complex(output_spec[:,:,:,0], output_spec[:,:,:,1])
    # Back to 2D matrix
    output_spec = tf.squeeze(output_spec, axis=[0])
    
    # Turning spectrogram back into .wav
    output = tf.signal.inverse_stft(output_spec, frame_length=512, frame_step=32, window_fn=tf.signal.inverse_stft_window_fn(32))
    normalize_ = tf.reduce_max(tf.abs(output))
    output = output / normalize_
    with open('output.wav', mode='w'):
        soundfile.write('output.wav', output, RATE_OUT)
    print('Done')


def make_spectrogram(wav):
    spectrogram = tf.signal.stft(wav, frame_length=512, frame_step=32)
    # Add dimensions for convolution
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    spectrogram = tf.expand_dims(spectrogram, axis=0)
    return spectrogram


def get_file(path):
    file_content = tf.io.read_file(path)
    wav , samplerate = tf.audio.decode_wav(file_content, desired_channels=2)
    # Mix both channels
    wav = wav[:,0] + wav[:,1] / 2
    samplerate = tf.cast(samplerate, dtype="int64")
    wav = tfio.audio.resample(wav, rate_in=samplerate, rate_out=RATE_OUT)
    return wav


def make_test(bg, fg):
    print('Making test')
    size_difference = bg.shape[0] - fg.shape[0]
    padding = tf.zeros(abs(size_difference), dtype=tf.float32)

    if size_difference > 0:
        fg = tf.concat([fg, padding], axis=0)
    else:
        bg = tf.concat([bg, padding], axis=0)

    test_ = bg * .3 + fg * .7
    normalize_ = tf.reduce_max(test_)
    test_ = test_ / normalize_

    with open('test.wav', mode='w'):
        soundfile.write('test.wav', test_, RATE_OUT)

main()
# %%
