# blogscraper.py
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os
import time
import re

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

client = OpenAI(api_key=OPENAI_API_KEY)

CHROMA_DB_PATH = "./chroma_db"
CHROMA_COLLECTION_NAME = "youtube_video_knowledge"  

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
try:
    collection = chroma_client.create_collection(name=CHROMA_COLLECTION_NAME)
except:
    collection = chroma_client.get_collection(name=CHROMA_COLLECTION_NAME)

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

BASE_URL = "https://www.nhs.uk/start-for-life/baby/recipes-and-meal-ideas/"
response = requests.get(BASE_URL)
soup = BeautifulSoup(response.text, "html.parser")

recipe_links = []
for a in soup.find_all("a", href=True):
    href = a['href']
    if "/start-for-life/baby/recipes-and-meal-ideas/" in href and href != BASE_URL:
        full_link = href if href.startswith("http") else "https://www.nhs.uk" + href
        if full_link not in recipe_links:
            recipe_links.append(full_link)

print(f"[INFO] Found {len(recipe_links)} recipe links")

all_chunks = []
for link in recipe_links:
    try:
        r = requests.get(link)
        recipe_soup = BeautifulSoup(r.text, "html.parser")

        title_tag = recipe_soup.find("h1")
        if not title_tag:
            continue
        title = title_tag.text.strip()

        ingredients, steps = [], []
        ingredients_header = recipe_soup.find("h2", string=lambda t: t and "Ingredients" in t)
        if ingredients_header:
            ul_tag = ingredients_header.find_next("ul", class_=False)
            if ul_tag:
                ingredients = [li.get_text(strip=True) for li in ul_tag.find_all("li")]

        method_header = recipe_soup.find("h2", string=lambda t: t and "Method" in t)
        if method_header:
            method_div = method_header.find_parent("div")
            if method_div:
                ol_tag = method_div.find("ol")
                if ol_tag:
                    steps = [p.get_text(strip=True) for p in ol_tag.find_all("p")]

        # Skip recipes without ingredients or steps
        if not ingredients or not steps:
            continue

        # Build recipe text
        recipe_text = f"{title}\n\nIngredients:\n" + "\n".join(ingredients)
        recipe_text += "\n\nInstructions:\n" + "\n".join(steps)
        recipe_text = clean_text(recipe_text)

        # Filter by keywords
        if not any(k.lower() in recipe_text.lower() for k in BABY_FOOD_KEYWORDS):
            continue

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". "]
        )
        chunks = splitter.split_text(recipe_text)
        all_chunks.extend(chunks)

        print(f"[INFO] Processed recipe: {title}, {len(chunks)} chunks")
        time.sleep(1)

    except Exception as e:
        print(f"[ERROR] Failed to process {link}: {e}")

# Generate embeddings and store
embeddings_list = []
for chunk in all_chunks:
    emb = client.embeddings.create(model="text-embedding-3-large", input=chunk)
    embeddings_list.append((chunk, emb.data[0].embedding))

for i, (chunk_text, vector) in enumerate(embeddings_list):
    collection.add(
        ids=[f"nhs_recipe_{i}"],
        documents=[chunk_text],
        embeddings=[vector]
    )

print("[INFO] All NHS recipes ingested successfully!")
