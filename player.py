import webbrowser

songName = "blue eyes"
query = songName.replace(' ', '+')
url = f"https://www.youtube.com/results?search_query={query}"

webbrowser.open(url)