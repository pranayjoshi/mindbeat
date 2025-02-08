# import neurokit2 as nk
# import numpy as np
# import pandas as pd
# from pylsl import StreamInlet, resolve_streams
# from scipy.signal import welch
# import time
# from brainflow.data_filter import DataFilter, FilterTypes, AggOperations

# # Function to compute power spectral density (PSD)
# def compute_psd(eeg_data, fs=256):
#     freqs, psd = welch(eeg_data, fs, nperseg=fs)
#     return freqs, psd

# # Function to extract EEG features using BrainFlow for noise removal
# def extract_features(eeg_data):
#     eeg_df = pd.DataFrame(eeg_data, columns=["TP9", "AF7", "AF8", "TP10"])
    
#     # Clean EEG signal using BrainFlow
#     eeg_cleaned = eeg_df.copy()
#     for column in eeg_cleaned.columns:
#         DataFilter.perform_bandpass(eeg_cleaned[column].values, 256, 1.0, 50.0, 4, FilterTypes.BUTTERWORTH.value, 0)
#         DataFilter.perform_bandstop(eeg_cleaned[column].values, 256, 50.0, 4.0, 4, FilterTypes.BUTTERWORTH.value, 0)
#         DataFilter.perform_bandstop(eeg_cleaned[column].values, 256, 60.0, 4.0, 4, FilterTypes.BUTTERWORTH.value, 0)

#     # Compute frequency power bands
#     psd_features = {}
#     for channel in eeg_cleaned.columns:
#         freqs, psd = compute_psd(eeg_cleaned[channel])
#         psd_features[channel] = {
#             "Delta": np.sum(psd[(freqs >= 0.5) & (freqs < 4)]),
#             "Theta": np.sum(psd[(freqs >= 4) & (freqs < 8)]),
#             "Alpha": np.sum(psd[(freqs >= 8) & (freqs < 12)]),
#             "Beta": np.sum(psd[(freqs >= 12) & (freqs < 30)]),
#             "Gamma": np.sum(psd[(freqs >= 30)])
#         }
    
#     return psd_features

# # Function to classify emotion based on EEG frequency bands
# def classify_emotion(psd_features):
#     # Average across all channels
#     avg_psd = {band: np.mean([ch[band] for ch in psd_features.values()]) for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]}

#     # Dictionary-based emotion mapping
#     emotions = {
#         "Happy": avg_psd["Alpha"] > avg_psd["Beta"] and avg_psd["Alpha"] > avg_psd["Theta"],
#         "Sad": avg_psd["Theta"] > avg_psd["Alpha"] and avg_psd["Theta"] > avg_psd["Beta"],
#         "Angry": avg_psd["Beta"] > avg_psd["Alpha"] and avg_psd["Beta"] > avg_psd["Theta"],
#         "Anxious": avg_psd["Gamma"] > avg_psd["Beta"],
#         "Concentration": avg_psd["Beta"] > avg_psd["Alpha"] and avg_psd["Beta"] > avg_psd["Theta"],
#         "Relaxed": avg_psd["Alpha"] > avg_psd["Beta"] and avg_psd["Alpha"] > avg_psd["Theta"],
#         "Drowsy": avg_psd["Theta"] > avg_psd["Alpha"] and avg_psd["Theta"] > avg_psd["Beta"],
#         "Deep Sleep": avg_psd["Delta"] > avg_psd["Theta"] and avg_psd["Delta"] > avg_psd["Alpha"],
#         "High-Level Focus": avg_psd["Gamma"] > avg_psd["Beta"]
#     }

#     # Determine the detected emotion
#     detected_emotion = "Neutral"
#     for emotion, condition in emotions.items():
#         if condition:
#             detected_emotion = emotion
#             break

#     return detected_emotion

# # Connect to Muse EEG stream
# print("Searching for Muse EEG stream...")
# streams = resolve_streams(wait_time=5.0)

# if not streams:
#     print("No Muse stream found. Ensure your device is connected and `muselsl stream` is running.")
#     exit()

# inlet = StreamInlet(streams[0])
# print("Muse stream found! Starting real-time emotion detection...")

# # Real-time EEG data collection
# while True:
#     try:
#         samples = []
#         timestamps = []

#         for _ in range(256):  # Collect 1 second of EEG data
#             sample, timestamp = inlet.pull_sample()
#             samples.append(sample[:4])  # First 4 channels: TP9, AF7, AF8, TP10
#             timestamps.append(timestamp)

#         eeg_data = np.array(samples)

#         # Preprocess EEG data and extract frequency features
#         psd_features = extract_features(eeg_data)

#         # Detect Emotion
#         emotion = classify_emotion(psd_features)

#         print(f"Detected Emotion: {emotion}")

#         # Sleep for 1 second before collecting the next batch
#         time.sleep(1)

#     except KeyboardInterrupt:
#         print("Stopping real-time emotion detection.")
#         break