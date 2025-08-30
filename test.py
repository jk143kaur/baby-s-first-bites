from main import load_conversation_chain  # import the function
import streamlit as st  # optional, only if you want to use st.write

# Load the chain and vector store
chain, vector_store = load_conversation_chain()

# Test retrieval from ChromaDB
retriever = vector_store.as_retriever()
query = "iron rich foods"  # example query
docs = retriever.get_relevant_documents(query)

print(f"[INFO] Retrieved {len(docs)} chunks for query: '{query}'")
if docs:
    print(docs[0].page_content[:500])  # preview first 500 chars
