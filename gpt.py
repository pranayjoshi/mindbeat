import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
API_KEY = os.getenv("KEY")
openai.api_key = API_KEY

# Store OpenAI client instance for external use
openai_client = openai

# Store previously recommended songs for each emotion
previous_recommendations = {}

# Define available emotions
emotions = [
    "Mildly Positive & Confident", "Slightly Positive but Hesitant", "Calm & Neutral", "Relaxed but Withdrawn", 
    "Frustrated but Assertive", "Stressed & Overwhelmed", "Indifferent & Passive", "Sad & Low Energy"
]

def get_song_recommendation(emotion):
    if emotion not in emotions:
        return "Invalid emotion. Please provide a valid emotion from the list."
    
    prompt = f"Suggest a song that matches the emotion: {emotion}. Provide only the song name and singer in the format 'Song, Singer'. Do not include any other text."
    
    response = openai_client.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "system", "content": "You are a helpful music recommendation assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=50
    )
    
    song = response["choices"][0]["message"]["content"].strip()
    
    # Ensure no duplicate song recommendations for the same emotion
    if emotion not in previous_recommendations:
        previous_recommendations[emotion] = set()
    
    while song in previous_recommendations[emotion]:
        response = openai_client.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful music recommendation assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=50
        )
        song = response["choices"][0]["message"]["content"].strip()
    
    previous_recommendations[emotion].add(song)
    return song

# Example usage
if __name__ == "__main__":
    user_emotion = "Mildly Positive & Confident"  # Change this to test other emotions
    print(get_song_recommendation(user_emotion))