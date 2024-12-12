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

def create_message_bubble(sentences_with_sentiment, is_speaker_a: bool):
    speaker_colors = {
        'A': '#E8F0FF',  # Light blue for speaker A
        'B': '#F0F0F0',  # Light gray for speaker B
    }
    
    # Create columns with overlapping widths
    col1, col2 = st.columns([8, 2])
    
    # Base bubble style
    bubble_style = f"""
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px;
        background-color: {speaker_colors['A'] if is_speaker_a else speaker_colors['B']};
        display: inline-block;
        width: 80%;
    """
    
    # Create markdown content with highlighted sentences
    markdown_content = ""
    for text, sentiment in sentences_with_sentiment:
        if sentiment == 'NEUTRAL':
            markdown_content += f"{text} "
        else:
            highlight_color = {
                'POSITIVE': '#98FB98',  # Pale green
                'NEGATIVE': '#FFB6C1',  # Light pink
            }[sentiment]
            markdown_content += f"<span style='background-color: {highlight_color}; padding: 2px 4px; border-radius: 3px;'>{text}</span> "
    
    # Add message to appropriate column
    if is_speaker_a:
        with col1:
            st.markdown(f"<div style='{bubble_style}'>{markdown_content}</div>", unsafe_allow_html=True)
    else:
        with col2:
            container_style = """
                margin-left: -400%;
                width: 500%;
            """
            st.markdown(
                f"<div style='{container_style}'><div style='text-align: right'>"
                f"<div style='{bubble_style}'>{markdown_content}</div></div></div>",
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
    
    # Group sentences by speaker
    current_speaker = None
    current_sentences = []
    
    for sent in sentiment_analysis:
        if current_speaker != sent.speaker:
            # Output accumulated sentences if speaker changes
            if current_sentences:
                create_message_bubble(
                    current_sentences,
                    current_speaker == 'A'
                )
                current_sentences = []
            
            current_speaker = sent.speaker
        
        # Add tuple of (text, sentiment) to current sentences
        current_sentences.append((sent.text, sent.sentiment.value))
    
    # Output final group
    if current_sentences:
        create_message_bubble(
            current_sentences,
            current_speaker == 'A'
        )

