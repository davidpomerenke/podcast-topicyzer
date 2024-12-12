import assemblyai as aai
from joblib import Memory
import streamlit as st
from collections import defaultdict
from typing import Literal

cache = Memory(".cache", verbose=0).cache

aai.settings.api_key = "ffb4aee947fd4de695dcb6ef6f44c54e"

config = aai.TranscriptionConfig(
    speaker_labels=True,
    sentiment_analysis=True,
)

@cache
def get_transcript(url):
    transcriber = aai.Transcriber(config=config)
    transcript = transcriber.transcribe(url)
    sentences = transcript.get_sentences()
    utterances = transcript.utterances
    sentiment_analysis = transcript.sentiment_analysis

    return sentences, utterances, sentiment_analysis

def create_message_bubble(text: str, is_speaker_a: bool):
    # Define colors for different speakers
    colors = {
        'A': '#E8F0FF',  # Light blue for speaker A
        'B': '#F0F0F0',  # Light gray for speaker B
    }
    
    # Create columns with overlapping widths (each takes 80% of space)
    col1, col2 = st.columns([8, 2])  # First column takes 80%
    
    # Style for the message bubble
    bubble_style = f"""
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px;
        background-color: {colors['A'] if is_speaker_a else colors['B']};
        display: inline-block;
        width: 80%;
    """
    
    # Add message to appropriate column
    if is_speaker_a:
        with col1:
            st.markdown(f"<div style='{bubble_style}'>{text}</div>", unsafe_allow_html=True)
    else:
        with col2:
            # Use negative margin to pull content back to the left
            container_style = """
                margin-left: -400%;
                width: 500%;
            """
            st.markdown(
                f"<div style='{container_style}'><div style='text-align: right'>"
                f"<div style='{bubble_style}'>{text}</div></div></div>",
                unsafe_allow_html=True
            )

st.set_page_config(page_title="Podcast Transcription Analyzer", page_icon=":speech_balloon:")
st.title("Podcast Transcription Analyzer")

# Input field for podcast URL
url = st.text_input(
    "Enter podcast URL",
    value="https://assembly.ai/weaviate-podcast-109.mp3"
)

if url:
    with st.spinner("Transcribing podcast..."):
        sentences, utterances, sentiment_analysis = get_transcript(url)

    for utterance in utterances:
        is_speaker_a = utterance.speaker == 'A'
        create_message_bubble(utterance.text, is_speaker_a)

