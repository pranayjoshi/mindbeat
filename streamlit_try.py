import streamlit as st

# Function to load HTML file
def load_html(file_path):
    with open(file_path, 'r') as file:
        html_content = file.read()
    return html_content

# Path to the HTML file
html_file_path = 'ex.html'

# Load the HTML content
html_content = load_html(html_file_path)

# Display the HTML content in Streamlit
st.markdown(html_content, unsafe_allow_html=True)