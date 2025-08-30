# Install dependencies if not already installed:
# pip install yt-dlp openai langchain chromadb tiktoken youtube-transcript-api

import argparse
import re
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

# -----------------------------
# Step 0: CLI Argument Parsing
# -----------------------------
parser = argparse.ArgumentParser(description="Ingest YouTube video transcript into ChromaDB")
parser.add_argument("--video_url", type=str, required=True, help="Full YouTube video URL to process")
args = parser.parse_args()

# Extract video ID from URL
match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", args.video_url)
if not match:
    raise ValueError("Invalid YouTube URL. Could not extract video ID.")
VIDEO_ID = match.group(1)

load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

CHROMA_COLLECTION_NAME = "youtube_video_knowledge"


# -----------------------------
# Step 1: Fetch transcript
# -----------------------------
transcript_api = YouTubeTranscriptApi()
transcript = transcript_api.fetch(VIDEO_ID)
transcript_text = " ".join([t.text for t in transcript])
print(f"[INFO] Transcript fetched for video {VIDEO_ID}.")

# -----------------------------
# Step 2: Split transcript into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_text(transcript_text)
print(f"[INFO] Transcript split into {len(chunks)} chunks.")

# -----------------------------
# Step 3: Generate embeddings
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
embeddings_list = []
for chunk in chunks:
    emb = client.embeddings.create(
        model="text-embedding-3-large",
        input=chunk
    )
    embeddings_list.append((chunk, emb.data[0].embedding))
print("[INFO] Embeddings generated for all chunks.")

# -----------------------------
# Step 4: Store chunks in ChromaDB
# -----------------------------

chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Create or get collection
try:
    collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME)
except:
    collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=OPENAI_API_KEY,
    model_name="text-embedding-3-large"
)

for i, (chunk_text, vector) in enumerate(embeddings_list):
    collection.add(
        ids=[f"chunk_{i}"],
        documents=[chunk_text],
        embeddings=[vector]
    )




print("[INFO] Chunks ingested into ChromaDB successfully.")
print("[INFO] ✅ LangSmith tracing enabled. Go to https://smith.langchain.com to view traces.")