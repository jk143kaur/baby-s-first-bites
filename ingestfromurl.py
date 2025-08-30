# ingestfromurl.py
import re
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
from chromadb.utils import embedding_functions
from youtube_transcript_api import YouTubeTranscriptApi
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_COLLECTION_NAME = "youtube_video_knowledge"

BABY_FOOD_KEYWORDS = [
    "baby", "infant", "solid food", "weaning", "puree", "mashed", 
    "breastfeeding", "formula", "cereal", "vegetables", "fruit", 
    "feeding", "nutrition"
]

def clean_text(text):
    """Remove unwanted metadata and clean whitespace"""
    text = re.sub(r"Metadata:.*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()

def ingest_video_from_url(video_url: str):
    # Extract video ID
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", video_url)
    if not match:
        raise ValueError("Invalid YouTube URL. Could not extract video ID.")
    VIDEO_ID = match.group(1)

    # Fetch transcript
    transcript = YouTubeTranscriptApi().fetch(VIDEO_ID)
    transcript_text = " ".join([t.text for t in transcript])
    transcript_text = clean_text(transcript_text)

    # Filter by baby food keywords
    if not any(keyword.lower() in transcript_text.lower() for keyword in BABY_FOOD_KEYWORDS):
        raise ValueError("Video does not appear to be related to baby food. Ingestion aborted.")

    # Split transcript into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". "]
    )
    chunks = splitter.split_text(transcript_text)

    # Generate embeddings
    client = OpenAI(api_key=OPENAI_API_KEY)
    embeddings_list = []
    for chunk in chunks:
        emb = client.embeddings.create(model="text-embedding-3-large", input=chunk)
        embeddings_list.append((chunk, emb.data[0].embedding))

    # Store in ChromaDB
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
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

    return len(chunks)
