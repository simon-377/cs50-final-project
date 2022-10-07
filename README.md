# **Noise Suppression AI**
### **Video Demo:**  <URL HERE>
### **Datasets: [LJ Speech](https://keithito.com/LJ-Speech-Dataset/), [UrbanSound8K](https://urbansounddataset.weebly.com/download-urbansound8k.html)**
----
### **Overview**
This is a general noise suppresion AI, it does not work that well but it does work. Put a *.wav* file into the folder and change the *WAV* constant at the top to your filepath. Make sure you don't forget the file extension. Now run the *eval.py* script and listen to *output.wav*.

If you want to run the noise.py script, you first need to download the two datasets. Then remove the non *.wav* files from the UrbanSound8K's foldX folders. This should only be one file each.

There is also a *make_PCM16.py* file that was used to convert the UrbanSound8K files into a format that tensorflow can handle. It was also used to determine the *BAD_FILES* constant that you can find at the top of *noise.py*. You will need to convert your file to *PCM_16* if it isn't already.

The suppression seems to mostly take out the lower frequencies which makes sense since the entire LJ Speech dataset is based on a woman, so the lower registers probably contain less information.

-------

### **Design**

First, lets talk about the **model** which is made in *noise.py*:

In main, we first read all the filepaths from the metadata.csv file into a list with *meta()*.

Then we call *data_cleanup()* on the data gotten from that. This function only concatenates the path of the directory with each specific file path. It used to count letters and replace some abbreviations and check the length of the wav files. The *letters* list is now just a bunch of place-holder values.

Now we *zip* the letters and paths together into a *dataset* after having converted it to tensors. *Tensorflow* *(Keras)* does not want python lists to be fed into its data-pipeline.

Now we build the pipeline: First we *shuffle*, this is particularly important since I kept having memory leaks which meant I could only run .3 of the *dataset* in a single epoch and then had to manually restart the training (I used checkpoints to save the progress). I was still able to use the whole *dataset* for training. I *shuffle* before map because else my 32GB's of RAM would be filled up a lot since we would have all the audio data in the *dataset* instead of just the file paths.

Now comes the biggest part, mapping the filepaths in the *dataset* onto audio data: We start by reading a speech file at the given location and converting it into a consistent samplerate using Tensorflow methods.

In order to get a random subsection with the length given in the *LENGTH* constant, we call a python function that has the audio tensor passed into it (hopefully by reference; this is almost certainly the case). Here we use the shape and *LENGTH* to give ourselves a random index.

Now we load a background audio file into a tensor by calling *load_background()*. Most of the steps are the same as before but at the beginning of the function, we call a python function *get_random_bg_path()* so that we can use some python libraries to get a random path from within the UrbanSound8K dataset.

This could definitely be rewritten so that we don't write the code to load a *wav* file twice.

Now, we set the label, which is what our model will try to reconstruct, as the audio tensor without any background noise and mix the speech with roughly 33% noise.

In order for the model to be able to recognize the difference between speech and background noise, we should probably give it an easy way to access frequencies. This can be done with a fourier transflorm which literally just transforms a bunch of waves into a frequency representation. The short-time-fourier-transform (*stft*) from Tensorflow has an inverse method which allows us to convert it back into normal audio data so it's an obvious choice.

I was following a [guide](https://www.youtube.com/watch?v=ZLIPkmmDJAc) and just copied the settings, I am still not entirely sure what *frame_length* does (only that it should probably be a power of two) but the *step_size* is probably just the resolution for how many time-steps are included (32 meaning we probably convert 32 audio samples into one pixel on the x-axis). The y-axis then gives us the frequencies present at that point. You would think we would lose a lot of audio quality with that *step_size* but it does not seem like it (I tested by just converting back and forth without modification).

In order to be able to convert the output of the model back into a *wav*, the *stft_inverse* wants a complex input (which the *stft* also does provide). The problem with this is that the model always gave me a real output even though the input and label where complex...

The layers that we end up using *(2DConvolution)* always take 4D inputs and give 4D outputs. The first dimension contains batch size (only 1 per batch in my case due to memory leak and no CUDA enabled GPU which makes a CPU bottleneck guaranteed anyway since that is the only thing actually being used), dimension 2, 3 is where the spectrogram goes, the last dimension is usually used for RGB values and would contain 3 entries per pixel in that case. We do not have RGB in our spectrogram.

We split the complex 64bit numbers from the *stft* into real and imaginary parts and concatenate them into the last dimension giving us two float32 entries per pixel. If we would use a single absolute value per pixel instead, the *stft_inverse* would make us sound like a chipmunk.

Now we are done mapping and the guide I followed also puts cache, batch and prefetch. I am not sure if those matter since training this on CPU is so slow anyway.

You might criticize the amount of python code used in *map* but I tested it and my computer spends the vast majority of the time working on the model and not loading data (else using smaller models wouldn't allow me to train very fast).

For the model I just figured I will have the lot: I used *2D convolution* and its transpose and just copied loads of layers until I felt like it was going to take a while to train. The transpose basically inverts the effect that the normal convolution has on the size of the tensor: If using a 3x3 kernel, we will lose 2 pixels on each spectrogram dimension per layer. This can be addressed with padding but I read that it is less efficient to do that. We use the same number of *transpose convolutions* as we use the normal version so we get the same output size as input size.

With a smaller model I had tried all sorts of settings and I found that larger kernel sizes just took forever to train and didn't yield a much better return. I decided on the small kernel and used lots of filters for lots of trainable parameters (33k), these did actually seem to matter. The last layer needs to have 2 filters (first positional argument) because these get written into the last dimension and we need 2 for reconstructing a complex number.

We set up loading and saving checkpoints and compile the model. Since I did not have time to read up on optimizers yet, I just copied the choice from the guide *(Adam)* and tried both *mean squared error* and *mean absolute error* as loss functions (both subtract the expected values in the label from the prediction by the model; then this is either squared to punish large deviations extra, or just kept absolute). I tried both and wasn't able to tell which one sounded better (using some kind of objective messure is a bit dangerous since audio is subjectively interpreted by us, so I relied on testing for this choice).

I decided in favor of the linear option (mean absolute error) since I didn't want to introduce any distortion in my incentive.

The model is now saved.

Now, let's talk about how this model is used to denoise a particular file in ***eval.py***:

There is a "make_test()" function that makes a testfile by mixxing a manually chosen speech with manually chosen background. The filepaths are chosen at the top of the script.

Beyond that, we just store the filepath of the wav that is supposed to be denoised in a constant *(WAV)* and perform pretty much the same steps as before to feed the data into our model. The only difference is that we obviously want the entire thing to be processed and not just a random *LENGTH* chunck.

For this we write a loop that continues to process one spectrogram at a time and concatenates the result into one large tensor. If it errors out *(ValueError)* it knows it's the end of the data and just takes the last chunk, it then cuts away the beginning of it so that it has the same size as the remainder. The chunks always need to be the same size with the *Sequential* model.

Now we take the last dimension with two entries per pixel and combine them into a single complex number each. We drop the unnecessary dimension and write it into *output.wav*.

This file used to be much more complex (see the following story section for more on that).

Maybe I needed to use a fully connected layer somewhere in there, maybe convolution only is too local. The result is not as good as I was hoping for.

--------

### **Story**
Initially, I wanted to make a chrome extension, that grabs the audio playing in a YouTube video, runs it through a *Tensorflow* based AI to count how quickly people are speaking and boosts the playback speed a bit if they are speeking slowly.

I had all three pieces: A self-made extension that could change and store playback speeds, a work around that I found on github that allowed me to get the audio playing in a tab (it did have an MIT license) and the ai that counted how quickly people are speeking (letters per second) which I also made myself.

Unfortunately I was not able to put these pieces together because Chrome does not let you use script tags in *html* files. This meant I had to bundle the *Tensorflow.js* library with my *JavaScript* file to bring it into the extension.

I tried that using *Extension CLI* but it did not recognize the kind of *JavaScript* that was used in *Tensorflow.js*. *Extension CLI* included *webpack* which apparently needed to be modified in some way for me to account for different flavors of *JavaScript*. I was not really able to do this and ended up having to pivot.

This is where I started working on a background noise suppresion model since it allowed me to at least keep my *Tensorflow* code since I was already combining speech with random background noise.

After some difficulties, I decided to make a categorizer model that finds all the places with background noise only (nobody speaking). This was then passed to another model which layers speech from the LJ Speech dataset over it and trains to remove it.

When I was done with that I re-decided that I should try a general noise suppresion again. This ended up actually kind of working since I had removed a lot of the issues that where due to the *map* method in *Tensorflow* not allowing me to write all kinds of python code in my data pipeline leading to it not working quite right while fitting even though when it was called seperately, it did work. Specifically the *random* library didn't work, I think it only rolled once at the beginning.

In the end the general noise suppression seems to work equally well as training on the particular background noise of the *wav* file.

The general noise suppresion actually has quite a bit less code, especially in the eval script. I hope it's still enough.