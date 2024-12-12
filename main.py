import assemblyai as aai
from joblib import Memory
import streamlit as st
from collections import defaultdict
from typing import Literal
import requests
import json
import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
import pandas as pd

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

@cache
def get_embeddings(texts: list[str]):
    url = "https://api.jina.ai/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer jina_7852c3b21d2c4da681894eb96fe8e4f0ayxfyJz8es_cUZXZOpn_GbeS0YM5",
    }
    data = {
        "model": "jina-embeddings-v3",
        "task": "text-matching",
        "late_chunking": False,
        "dimensions": 1024,
        "embedding_type": "float",
        "input": texts,
    }
    response = requests.post(url, headers=headers, json=data)
    result = json.loads(response.text)
    return np.array(result['data'][0]['embedding'])


def create_message_bubble(sentences_with_sentiment, is_speaker_a: bool):
    speaker_colors = {
        "A": "#E8F0FF",  # Light blue for speaker A
        "B": "#F0F0F0",  # Light gray for speaker B
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
        if sentiment == "NEUTRAL":
            markdown_content += f"{text} "
        else:
            highlight_color = {
                "POSITIVE": "#98FB98",  # Pale green
                "NEGATIVE": "#FFB6C1",  # Light pink
            }[sentiment]
            markdown_content += f"<span style='background-color: {highlight_color}; padding: 2px 4px; border-radius: 3px;'>{text}</span> "

    # Add message to appropriate column
    if is_speaker_a:
        with col1:
            st.markdown(
                f"<div style='{bubble_style}'>{markdown_content}</div>",
                unsafe_allow_html=True,
            )
    else:
        with col2:
            container_style = """
                margin-left: -400%;
                width: 500%;
            """
            st.markdown(
                f"<div style='{container_style}'><div style='text-align: right'>"
                f"<div style='{bubble_style}'>{markdown_content}</div></div></div>",
                unsafe_allow_html=True,
            )


st.set_page_config(
    page_title="Podcast Transcription Analyzer", page_icon=":speech_balloon:"
)
st.title("Podcast Transcription Analyzer")

# Input field for podcast URL
url = st.text_input(
    "Enter podcast URL", value="https://assembly.ai/weaviate-podcast-109.mp3"
)

if url:
    with st.spinner("Transcribing podcast..."):
        sentences, utterances, sentiment_analysis = get_transcript(url)

    # Group sentences by speaker
    current_speaker = None
    current_sentences = []
    all_texts = []
    speakers = []
    sentiments = []

    for sent in sentiment_analysis:
        if current_speaker != sent.speaker:
            if current_sentences:
                create_message_bubble(current_sentences, current_speaker == "A")
                # Store the combined text for embedding
                all_texts.append(" ".join(text for text, _ in current_sentences))
                speakers.append(current_speaker)
                # Use the most common sentiment in the bubble
                sentiment_counts = defaultdict(int)
                for _, sent_value in current_sentences:
                    sentiment_counts[sent_value] += 1
                sentiments.append(max(sentiment_counts.items(), key=lambda x: x[1])[0])
                current_sentences = []

            current_speaker = sent.speaker

        current_sentences.append((sent.text, sent.sentiment.value))

    # Output final group
    if current_sentences:
        create_message_bubble(current_sentences, current_speaker == "A")
        all_texts.append(" ".join(text for text, _ in current_sentences))
        speakers.append(current_speaker)
        sentiment_counts = defaultdict(int)
        for _, sent_value in current_sentences:
            sentiment_counts[sent_value] += 1
        sentiments.append(max(sentiment_counts.items(), key=lambda x: x[1])[0])

    # Get embeddings and create visualization
    if all_texts:
        with st.spinner("Generating topic visualization..."):
            # Get embeddings for all text bubbles
            embeddings = np.array([get_embeddings([text]) for text in all_texts])
            
            # Create clusters of 10 messages (instead of 6)
            cluster_size = 10
            n_clusters = len(all_texts) // cluster_size + (1 if len(all_texts) % cluster_size else 0)
            
            clustered_embeddings = []
            clustered_speakers = []
            clustered_numbers = []
            
            for i in range(n_clusters):
                start_idx = i * cluster_size
                end_idx = min(start_idx + cluster_size, len(all_texts))
                
                # Average the embeddings for this cluster
                cluster_embedding = embeddings[start_idx:end_idx].mean(axis=0)
                clustered_embeddings.append(cluster_embedding)
                
                # Get most frequent speaker in this cluster
                cluster_speakers = speakers[start_idx:end_idx]
                most_common_speaker = max(set(cluster_speakers), key=cluster_speakers.count)
                clustered_speakers.append(most_common_speaker)
                
                # Store the range of message numbers
                clustered_numbers.append(f"{start_idx+1}-{end_idx}")
            
            # Convert to numpy array for t-SNE
            clustered_embeddings = np.array(clustered_embeddings)
            
            # Reduce to 2D using t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(clustered_embeddings)-1))
            points_2d = tsne.fit_transform(clustered_embeddings)

            # Create DataFrame for plotting
            df = pd.DataFrame({
                'x': points_2d[:, 0],
                'y': points_2d[:, 1],
                'Speaker': clustered_speakers,
                'Messages': clustered_numbers
            })

            # Create interactive plot
            fig = px.scatter(
                df,
                x='x',
                y='y',
                text='Messages',
                title='Conversation Flow: Topic Evolution (Grouped by 10 Messages)',
                labels={'x': 'Topic Dimension 1', 'y': 'Topic Dimension 2'},
            )
            
            # Customize the appearance
            fig.update_traces(
                textposition='top center',
                textfont=dict(size=14),
                marker=dict(size=20, color='#636EFA'),  # Single color for all points
            )
            
            # Add arrows to show conversation flow
            for i in range(len(df) - 1):
                fig.add_shape(
                    type='line',
                    x0=df.iloc[i]['x'],
                    y0=df.iloc[i]['y'],
                    x1=df.iloc[i + 1]['x'],
                    y1=df.iloc[i + 1]['y'],
                    line=dict(color='gray', width=2, dash='dot'),
                    opacity=0.7
                )

            st.plotly_chart(fig)
