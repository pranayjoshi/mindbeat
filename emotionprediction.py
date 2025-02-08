from io import StringIO
import sys
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from pylsl import StreamInlet, resolve_byprop
from scipy import signal, stats
import mne
import time

class PrintToStreamlit:
    def __init__(self):
        self.buffer = StringIO()

    def write(self, text):
        self.buffer.write(text)

    def flush(self):
        pass

    def get_logs(self):
        return self.buffer.getvalue()

# Initialize the custom print class
print_capture = PrintToStreamlit()
sys.stdout = print_capture

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




def iterEEG(inlet, plot_placeholder, logs_placeholder, final_plot_placeholder):
    for i in range(4):
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

        print("Got {} samples in {:.5f} seconds".format(len(data), t))
        logs_placeholder.text(print_capture.get_logs()) 
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
    final_plot_placeholder.pyplot(ax)

    print("Final Plot Plotted")

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

def main():
    st.title("EEG Emotion Prediction with BCI")
    st.write("This app predicts emotions based on EEG data from a MUSE 2 headset.")
    # Sidebar for logs
    st.sidebar.title("Logs")
    logs_placeholder = st.sidebar.empty()

    # Load model
    try:
        model_path = "EmotionNetV2.h5"
        for test_samples in [256, 384, 512, 768]:
            print(f"\nTrying with desiredSamples = {test_samples}")
            try:
                model = load_h5_model(model_path, test_samples)
                print(f"Success with desiredSamples = {test_samples}")
                break
            except Exception as e:
                print(f"Failed with {test_samples} samples")
                continue

    except Exception as e:
        print(f"Error: {str(e)}")

    # Connect to EEG stream
    print('Looking for an EEG stream...')
    logs_placeholder.text(print_capture.get_logs()) 
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')
    print("Start acquiring data")
    logs_placeholder.text(print_capture.get_logs()) 
    inlet = StreamInlet(streams[0], max_buflen=60, max_chunklen=int(inputLength))
    eeg_time_correction = inlet.time_correction()
    info = inlet.info()
    description = info.desc()
    fs = int(info.nominal_srate())

    # Print stream info placeholders for iterator
    print("Stream connected.")
    print("Sampling frequency: {} points per second".format(fs))
    print("Processing Muse EEG data...\n")
    logs_placeholder.text(print_capture.get_logs()) 
    plot_placeholder = st.empty()
    final_plot_placeholder = st.empty()
    iterEEG(inlet, plot_placeholder, logs_placeholder, final_plot_placeholder=final_plot_placeholder)
    fs = int(info.nominal_srate())
    print("Sampling frequency: {} Hz".format(fs))


    # Final Emotion Detection
    try:
        # preprocess EEG data
        data, timestamp = inlet.pull_chunk(timeout=5, max_samples=samples)
        eeg = np.array(data).swapaxes(0,1)
        processedEEG = signal.resample(eeg, int(eeg.shape[1] * (128 / fs)), axis=1)
        processedEEG = mne.filter.filter_data(processedEEG, sfreq=128, l_freq=4, h_freq=45, 
                                            filter_length='auto', l_trans_bandwidth='auto', 
                                            h_trans_bandwidth='auto', method='fir', 
                                            phase='zero', fir_window='hamming', verbose=0)
        processedEEG -= np.mean(processedEEG, axis=1, keepdims=True)
        print(processedEEG)
        for channel in range(buffers.shape[0]):
            buffers[channel] = updateBuffer(buffers[channel], processedEEG[channel])
        chunk_size = 256
        num_chunks = buffers.shape[1] // chunk_size


        # Store predictions
        emotion_predictions = []
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size

            # Extract chunk and reshape for the model
            input_data = buffers[:, start:end]  # Shape (4, 256)
            input_data = np.expand_dims(input_data, axis=0)  # Shape (1, 4, 256)

            # Predict and log emotions for this chunk
            emotions = model.predict(input_data)
            print("Emotions:", emotions)
            emotions = scale_emotions(emotions)
            print("Raw emotions:", emotions)
            print("Scaled emotions:", emotions)
            logs_placeholder.text(print_capture.get_logs())

            if emotions.ndim == 1:
                emotions = emotions.reshape(1, -1)
            elif emotions.shape[-1] != 3:
                print("Unexpected model output shape:", emotions.shape)
                continue
            emotions = np.clip(emotions, 1, 9)
            emotion_predictions.append(emotions[0])

        # Ensure we have valid predictions
            mean_emotions = np.mean(emotion_predictions, axis=0)

            valence = mean_emotions[0]
            arousal = mean_emotions[1]
            dominance = mean_emotions[2]

        print("Final Emotion Averages:")
        print("Valence: {:.2f}".format(valence))
        print("Arousal: {:.2f}".format(arousal))
        print("Dominance: {:.2f}".format(dominance))
        logs_placeholder.text(print_capture.get_logs()) 

        # Determine mood
        end_mood = determine_mood(valence, arousal, dominance)
        print("End Mood:", end_mood)


        logs_placeholder.text(print_capture.get_logs()) 
        st.markdown(f"## End Mood: **{end_mood}**")
        st.markdown(f"### Valence: **{valence:.2f}**")
        st.markdown(f"### Arousal: **{arousal:.2f}**")
        st.markdown(f"### Dominance: **{dominance:.2f}**")
    finally:
        
        
        inlet.close_stream()
        print("Stream closed.")

if __name__ == "__main__":
    main()