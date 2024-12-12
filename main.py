import assemblyai as aai
from joblib import Memory

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


sentences, utterances, sentiment_analysis = get_transcript("https://assembly.ai/weaviate-podcast-109.mp3")

print(sentences)
print(utterances)
print(sentiment_analysis)
