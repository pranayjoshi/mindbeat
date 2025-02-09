from matplotlib import pyplot as plt
import numpy as np
import time
import mne
from scipy import signal
import tensorflow as tf
from tensorflow import keras

def calculate_output_shape(input_shape, model):
    """
    Print the output shape of each layer to debug dimension issues
    """
    x = tf.zeros((1,) + input_shape)
    for layer in model.layers:
        x = layer(x)
        # print(f"{layer.name}: output_shape = {x.shape}")
    return x.shape

def load_h5_model(model_path, desiredSamples):
    """
    Load H5 model with shape debugging
    """

    # Create model
    model = keras.Sequential([
        # Input layer
        keras.layers.InputLayer(input_shape=(4, desiredSamples)),
        keras.layers.Reshape((4, desiredSamples, 1), name='reshape'),

        # First Conv Block
        keras.layers.Conv2D(32, (1, 4), strides=(1,1), activation='relu', padding='same', name='conv1'),
        keras.layers.MaxPooling2D((1, 2), name='pool1'),

        # Second Conv Block
        keras.layers.Conv2D(64, (1, 8), strides=(1,1), activation='relu', padding='same', name='conv2'),
        keras.layers.MaxPooling2D((1, 2), name='pool2'),

        # Third Conv Block
        keras.layers.Conv2D(64, (1, 8), strides=(1,1), activation='relu', padding='same', name='conv3'),
        keras.layers.MaxPooling2D((1, 2), name='pool3'),

        # Fourth Conv Block
        keras.layers.Conv2D(64, (1, 64), strides=(1,1), activation='relu', padding='same', name='conv4'),
        keras.layers.MaxPooling2D((1, 2), name='pool4'),

        # Fifth Conv Layer
        keras.layers.Conv2D(64, (1, 8), strides=(1,1), activation='relu', padding='same', name='conv5'),

        # Output layers
        keras.layers.Flatten(name='flatten'),
        keras.layers.Dense(64, activation='relu', name='dense1'),
        keras.layers.Dense(3, activation='linear', name='dense2')
    ])

    final_shape = calculate_output_shape((4, desiredSamples), model)
    try:
        # Try loading weights
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

        print("\nModel loaded successfully!")
        return model
    except Exception as e:
        print(f"\nDetailed error: {str(e)}")
        raise Exception("Failed to load H5 model")

#
fs = 256
inputLength = 10.5 # Length of input in seconds
shiftLength = 5 # Time between epochs
samples = int(shiftLength * fs) # How many samples to gather in every cycle
# print(samples)

bufferSize = int(128 * inputLength) # Size of buffer in samples. Enough to hold one set of downsampled input.

buffers = np.zeros((4, bufferSize)) # buffers for each of the four channels

# Push new data onto buffer, removing any old data on the end
def updateBuffer(buffer, newData):
    assert len(newData.shape) == len(buffer.shape) and buffer.shape[0] >= newData.shape[0], "Buffer shape ({}) and new data shape ({}) are not compatible.".format(buffer.shape, newData.shape)
    size = newData.shape[0]
    buffer[:-size] = buffer[size:]
    buffer[-size:] = newData
    return buffer
# Get the streamed data from the Muse. Blue Muse must be streaming.




def iterEEG(inlet, plot_placeholder, logs_placeholder):
    for i in range(5):
        start = time.time()
        data, timestamp = inlet.pull_chunk(timeout=5, max_samples=samples)
        t = time.time() - start
        eeg = np.array(data).swapaxes(0,1)

        # Downsample
        processedEEG = signal.resample(eeg, int(eeg.shape[1] * (128 / fs)), axis=1)

        # Apply bandpass filter from 4-45Hz
        processedEEG = mne.filter.filter_data(processedEEG, sfreq=128, l_freq=4, h_freq=45, 
                                        filter_length='auto', l_trans_bandwidth='auto', 
                                        h_trans_bandwidth='auto', method='fir', 
                                        phase='zero', fir_window='hamming', verbose=0)

        # Zero mean
        processedEEG -= np.mean(processedEEG, axis=1, keepdims=True)
        if i == 0:
            continue
        # Update buffer
        for channel in range(buffers.shape[0]):
            buffers[channel] = updateBuffer(buffers[channel], processedEEG[channel])

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(buffers[0], label="Tp9")
        ax.plot(buffers[1], label="AF7")
        ax.plot(buffers[2], label="AF8")
        ax.plot(buffers[3], label="Tp10")
        ax.legend()
        plot_placeholder.pyplot(fig)

    print("Record Brain waves Iteration end!")
    ax, plti  = plt.subplots(figsize=(10, 5))
    plti.plot(buffers[0,:256], label="Tp9")
    plti.plot(buffers[1,:256], label="AF7")
    plti.plot(buffers[2,:256], label="AF8")
    plti.plot(buffers[3,:256], label="Tp10")
    plti.legend()
    plot_placeholder.pyplot(ax)

    print("Final Plot Plotted")
    return ax

def determine_mood(valence, arousal, dominance):
    if valence >= 5 and arousal >= 5:
        if dominance >= 5:
            return "Mildly Positive & Confident"
        else:
            return "Slightly Positive but Hesitant"
    elif valence >= 5 and arousal < 5:
        if dominance >= 5:
            return "Calm & Neutral"
        else:
            return "Relaxed but Withdrawn"
    elif valence < 5 and arousal >= 5:
        if dominance >= 5:
            return "Frustrated but Assertive"
        else:
            return "Stressed & Overwhelmed"
    else:  # valence < 5 and arousal < 5
        if dominance >= 5:
            return "Indifferent & Passive"
        else:
            return "Sad & Low Energy"



# Example scaling function
def scale_emotions(emotions, target_min=1, target_max=9):
    source_min, source_max = -1, 1
    scaled_emotions = (emotions - source_min) * (target_max - target_min) / (source_max - source_min) + target_min
    return scaled_emotions