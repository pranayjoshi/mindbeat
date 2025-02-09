import openai
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
API_KEY = os.getenv("KEY")
openai.api_key = API_KEY

# Store OpenAI client instance
openai_client = openai

# Store previously recommended songs for each emotion
previous_recommendations = {}

# Define available emotions
emotions = [
    "Mildly Positive & Confident", "Slightly Positive but Hesitant", "Calm & Neutral", 
    "Relaxed but Withdrawn", "Frustrated but Assertive", "Stressed & Overwhelmed", 
    "Indifferent & Passive", "Sad & Low Energy"
]

def get_song_recommendation(emotion):
    if emotion not in emotions:
        return "Invalid emotion. Please provide a valid emotion from the list."
    
    prompt = (
        f'Generate a list of exactly 10 songs that match the emotion: "{emotion}". '
        "Each song must be suitable for a semi-professional setting. "
        "Ensure that all 10 slots are filled. Format your response strictly as follows:\n\n"
        "1. Song, Singer\n"
        "2. Song, Singer\n"
        "3. Song, Singer\n"
        "4. Song, Singer\n"
        "5. Song, Singer\n"
        "6. Song, Singer\n"
        "7. Song, Singer\n"
        "8. Song, Singer\n"
        "9. Song, Singer\n"
        "10. Song, Singer\n\n"
        "Provide no additional text, explanations, or numbering deviations. "
        "Do not return fewer than 10 songs under any circumstances."
    )

    response = openai_client.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are a helpful music recommendation assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=200  # Increase to ensure full output
    )

    song_list = response["choices"][0]["message"]["content"].strip().split("\n")
    song_list = [song.strip().split(". ", 1)[-1] for song in song_list]  # Remove numbering

    # Ensure no duplicate song recommendations for the same emotion
    if emotion not in previous_recommendations:
        previous_recommendations[emotion] = set()

    # Filter out previously recommended songs
    new_songs = [song for song in song_list if song not in previous_recommendations[emotion]]

    # If all songs are duplicates, regenerate
    while len(new_songs) == 0:
        response = openai_client.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a helpful music recommendation assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=200
        )
        song_list = response["choices"][0]["message"]["content"].strip().split("\n")
        song_list = [song.strip().split(". ", 1)[-1] for song in song_list]
        new_songs = [song for song in song_list if song not in previous_recommendations[emotion]]

    # Add new songs to the recommendation history
    previous_recommendations[emotion].update(new_songs)

    return new_songs  # Returns the list of songs instead of one

# Example usage
if __name__ == "__main__":
    user_emotion = "Mildly Positive & Confident"  # Change this to test other emotions
    recommendations = get_song_recommendation(user_emotion)
    print("\n".join(recommendations))