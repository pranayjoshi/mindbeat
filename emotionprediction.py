from io import StringIO
import sys
from time import sleep
from gpt import get_song_recommendation
import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
from pylsl import StreamInlet, resolve_byprop
from scipy import signal, stats
from emotionutils import load_h5_model, updateBuffer, scale_emotions, determine_mood, iterEEG
import mne


fs = 256
inputLength = 10.5 # Length of input in seconds
shiftLength = 5 # Time between epochs
samples = int(shiftLength * fs) # How many samples to gather in every cycle
# print(samples)

bufferSize = int(128 * inputLength) # Size of buffer in samples. Enough to hold one set of downsampled input.

buffers = np.zeros((4, bufferSize)) # buffers for each of the four channels

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

# Page configuration
st.set_page_config(
    page_title="AI Lyric Visualizer",
    page_icon="🎵",
    layout="wide",
)

# Initialize session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'home'

def navigate_to_dashboard():
    st.session_state.page = 'dashboard'

def navigate_to_playlist():
    st.session_state.page = 'playlist'

def Home():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

            .main {
                background-color: #0A0A0A;
                color: #FFFFFF;
                font-family: 'Inter', sans-serif;
            }

            .top-left {
                position: absolute;
                top: 20px;
                left: 20px;
                display: flex;
                align-items: center;
            }
            .top-left img {
                height: 60px;
                weight: 60px;
                margin-bottom: 20px;
                margin-right: 10px;
            }
            .top-left .name {
                font-size: 1.5rem;
                font-weight: 700;
                color: #FFFFFF;
            }

            .center-container {
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                min-height: calc(70vh - 100px);
                padding: 2rem;
                text-align: center;
                position: relative;
                overflow: hidden;
            }

            .header2 {
                font-size: 3.8rem;
                font-weight: 800;
                background: linear-gradient(90deg, #9333E0, #C026D3);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 1.5rem;
            }

            .subheader {
                font-size: 1.2rem;
                color: #9CA3AF;
                margin-bottom: 2rem;
                max-width: 48rem;
                line-height: 1.5;
            }

            .floating-element {
                position: absolute;
                background-color: rgba(147, 51, 234, 0.3);
                border-radius: 50%;
                animation: float 15s infinite;
            }

            @keyframes float {
                0% { transform: translate(0, 0) rotate(0deg); }
                33% { transform: translate(30px, -50px) rotate(120deg); }
                66% { transform: translate(-20px, 20px) rotate(240deg); }
                100% { transform: translate(0, 0) rotate(360deg); }
            }

            /* Align Streamlit button */
            div.stButton > button {
                background: linear-gradient(90deg, #9333EA 0%, #C026D3 100%);
                color: white;
                border-radius: 8px;
                padding: 0.75rem 2rem;
                border: none;
                font-size: 1.125rem;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s ease;
                text-decoration: none;
                display: inline-block;
                margin-top: 10px;
                box-shadow: 0 4px 14px rgba(147, 51, 234, 0.3);
            }
            div.stButton > button:hover {
                opacity: 0.9;
                transform: translateY(-2px);
            }
        </style>
        
        <div class="top-left">
            <img src="https://github.com/pranayjoshi/mind_music/blob/main/log.png?raw=true">
            <div class="header2">MindBeats</div>
        </div>
        
        <div class="center-container">
            <div class="floating-element" style="width: 100px; height: 100px; top: 10%; left: 10%;"></div>
            <div class="floating-element" style="width: 50px; height: 50px; top: 20%; right: 20%;"></div>
            <div class="floating-element" style="width: 75px; height: 75px; bottom: 15%; left: 15%;"></div>
            <div class="header2">Neuroadaptive Music Recommendation</div>
            <div class="subheader">Experience the future of music with MindBeats.
            Our BCI headset reads your emotions in real-time, creating perfectly curated playlists that match your mood.
            As your emotions evolve, our AI adapts your music dynamically, learning your preferences to deliver an ever-improving, personalized listening experience.</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Place the button right below the content
    col1, col2, col3 = st.columns([3, 2, 2])
    with col2:
        if st.button("Get Started ⚡"):
            navigate_to_dashboard()


def dashboard():

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
    iterEEG(inlet, plot_placeholder, logs_placeholder)
    fs = int(info.nominal_srate())
    print("Sampling frequency: {} Hz".format(fs))
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
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### Valence: **{valence:.2f}**")
    with col2:
        st.markdown(f"### Arousal: **{arousal:.2f}**")
    with col3:
        st.markdown(f"### Dominance: **{dominance:.2f}**")

    recommendations = get_song_recommendation(end_mood)
    print("\n".join(recommendations))
    logs_placeholder.text(print_capture.get_logs())
    # Add a button to navigate to the playlist page
    inlet.close_stream()
    print("Stream closed.")
    logs_placeholder.text(print_capture.get_logs())
    sleep(2)
    st.markdown("<div style='height: 200px;'></div>", unsafe_allow_html=True)  # Add space before the playlist
    st.title("Your Playlist")
    st.write("Here are your recommended songs:")

    # Example list of songs with Spotify and YouTube links
    songs = recommendations

    st.markdown(
        """
        <style>
            .playlist-container {
                padding-top: 50px;
            }
            .playlist-table {
                width: 100%;
                border-collapse: collapse;
            }
            .playlist-table th, .playlist-table td {
                border: 1px solid #ddd;
                padding: 8px;
                text-align: center;
            }
            .playlist-table th {
                padding-top: 12px;
                padding-bottom: 12px;
                background-color: #4CAF50;
                color: white;
            }
            .playlist-table td a {
                text-decoration: none;
                color: white;
            }
            .playlist-table td button {
                background: linear-gradient(90deg, #1DB954 0%, #1DB954 100%);
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .playlist-table td button:hover {
                opacity: 0.9;
            }
            .playlist-table td button.youtube {
                background: linear-gradient(90deg, #FF0000 0%, #FF0000 100%);
            }
            .playlist-table td.song-title, .playlist-table td.actions {
                width: 50%;
            }
        </style>
        <table class="playlist-table">
        """,
        unsafe_allow_html=True
    )

    for song in songs:
        st.markdown(
            f"""
            <tr>
                <td class="song-title">{song}</td>
                <td class="actions">
                    <a href="" target="_blank">
                        <button class="custom-button">Spotify</button>
                    </a>
                    <a href="" target="_blank">
                        <button class="custom-button youtube">YouTube</button>
                    </a>
                </td>
            </tr>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</table>", unsafe_allow_html=True)


if st.session_state.page == 'home':
    Home()
elif st.session_state.page == 'dashboard':
    dashboard()