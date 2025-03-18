import streamlit as st
import whisper
import numpy as np
import pickle
import chromadb
from sentence_transformers import SentenceTransformer
import tempfile
import os
from pydub import AudioSegment

# Load the subtitle embeddings (Already Generated)
def load_subtitle_embeddings():
    with open("subtitle_embeddings.pkl", "rb") as f:
        return pickle.load(f)

# Initialize ChromaDB for fast search
def init_chroma_db():
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="subtitles")

# Load Whisper Model
def load_whisper_model():
    return whisper.load_model("base")

# Load Sentence Transformer Model for Semantic Search
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Transcribe Audio using Whisper Model
def transcribe_audio(audio_file, whisper_model):
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    with open(temp_path, "wb") as f:
        f.write(audio_file.getbuffer())

    # Convert MP3 to WAV format as Whisper requires WAV
    sound = AudioSegment.from_file(temp_path)
    wav_path = temp_path.replace(".mp3", ".wav")
    sound.export(wav_path, format="wav")

    result = whisper_model.transcribe(wav_path, word_timestamps=True)
    os.remove(temp_path)
    os.remove(wav_path)

    return format_subtitles(result["segments"])

# Format time into SRT compatible format
def format_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

# Convert transcribed segments into subtitles format
def format_subtitles(segments):
    subtitles = []
    for i, segment in enumerate(segments):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        start_time_str = format_time(start_time)
        end_time_str = format_time(end_time)

        subtitles.append(f"{i+1}\n{start_time_str} --> {end_time_str}\n{text}\n")

    return subtitles

# Search for subtitles using Semantic Search
def search_subtitles(query, embedder, collection):
    query_embedding = embedder.encode(query).tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=5)
    return results["documents"][0] if results["documents"] else []

# Streamlit UI
def streamlit_ui():
    st.title("🎬 Subtitle Search Engine (Cloning Shazam)")

    # Upload Audio File
    uploaded_audio = st.file_uploader("🔊 Upload an audio clip (MP3, WAV, etc.)", type=["mp3", "wav"])

    if uploaded_audio:
        st.audio(uploaded_audio, format="audio/mp3")

        # Button to trigger audio transcription and show results
        if st.button("Transcribe Audio"):
            subtitle_list = transcribe_audio(uploaded_audio, whisper_model)

            if subtitle_list:
                st.write("📜 *Generated Subtitles:*")
                for subtitle in subtitle_list:
                    st.text(subtitle)

                # Convert subtitles to SRT format for download
                subtitle_srt = "\n".join(subtitle_list)
                subtitle_filename = "generated_subtitles.srt"

                with open(subtitle_filename, "w", encoding="utf-8") as f:
                    f.write(subtitle_srt)

                with open(subtitle_filename, "rb") as f:
                    st.download_button(label="📥 Download Subtitles (.srt)", data=f, file_name="subtitles.srt", mime="text/plain")
            else:
                st.write("❌ No subtitles generated.")

if __name__ == "__main__":
    # Load models and resources
    subtitle_data = load_subtitle_embeddings()
    chroma_collection = init_chroma_db()
    whisper_model = load_whisper_model()
    embedder = load_embedder()

    # Run the Streamlit UI
    streamlit_ui()
