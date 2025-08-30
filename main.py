# main.py
import streamlit as st
import os
import openai
from langchain_openai import OpenAI as LangchainOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from streamlit_mic_recorder import mic_recorder
from pathlib import Path
from ingestfromurl import ingest_video_from_url

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

PERSIST_DIRECTORY = "chroma_db"
SPEECH_FILE_PATH = Path(__file__).parent / "speech.mp3"
CHROMA_COLLECTION_NAME = "youtube_video_knowledge"

custom_template = """
You are a helpful and knowledgeable baby food assistant.

When the user asks a recipe-related question (e.g. "How to make X?"),
search the provided context for the most relevant recipe.

Your answer **must include**:
- Recipe name
- Age recommendation
- Preparation time
- Ingredients (as a bullet list)
- Step-by-step instructions

DO NOT use the word "metadata" in your reply.
If the context does not contain the recipe, politely say you don’t know.

Context: {context}
Chat History: {chat_history}
Question: {question}
Answer:
"""
CUSTOM_PROMPT = ChatPromptTemplate.from_template(custom_template)


def transcribe_audio(audio_bytes: bytes) -> str:
    try:
        with open("temp_audio.wav", "wb") as f:
            f.write(audio_bytes)
        with open("temp_audio.wav", "rb") as audio_file:
            transcript = openai.audio.transcriptions.create(model="whisper-1", file=audio_file)
        return transcript.text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""


def generate_and_play_audio(text: str):
    try:
        response = openai.audio.speech.create(model="tts-1", voice="alloy", input=text)
        response.stream_to_file(SPEECH_FILE_PATH)
        st.audio(str(SPEECH_FILE_PATH), autoplay=True)
    except Exception as e:
        st.error(f"Error generating or playing audio: {e}")


def load_conversation_chain():   #RAG
    """Return (chain, vector_store)"""
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        vector_store = Chroma(
            persist_directory=PERSIST_DIRECTORY,
            embedding_function=embeddings,
            collection_name=CHROMA_COLLECTION_NAME
        )
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        chain = ConversationalRetrievalChain.from_llm(
            llm=LangchainOpenAI(temperature=0),
            retriever=vector_store.as_retriever(),
            memory=memory,
            combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT}
        )
        return chain, vector_store
    except Exception as e:
        st.error(f"Error initializing the conversation chain: {e}")
        return None, None


def main():
    st.set_page_config(page_title="Baby's First Bites", page_icon="👶", layout="wide")

    st.sidebar.header("Upload Your Own Video 📹")
    with st.sidebar.expander("Ingest a new YouTube video"):
        user_video_url = st.text_input("Enter YouTube video URL:")
        if st.button("Ingest Video"):
            if user_video_url:
                with st.spinner("Ingesting video, please wait..."):
                    try:
                        chunks_added = ingest_video_from_url(user_video_url)
                        st.success(f"Video ingested successfully! {chunks_added} chunks added.")
                        # re-load chain & vector store and save them separately
                        st.session_state.chain, st.session_state.vector_store = load_conversation_chain()
                    except Exception as e:
                        st.error(f"Ingestion failed: {e}")
            else:
                st.warning("Please enter a valid YouTube URL.")

    # header
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("https://em-content.zobj.net/source/microsoft-teams/337/baby_1f476.png", width=150)
    with col2:
        st.title("Baby's First Bites 👶")
        st.markdown("""
        **Hello! I'm here to help you with baby food recipes and tips from trusted sources.**
        Ask me anything about solid foods, ingredients, or preparation based on the videos and recipes I've learned from.
        """)
    st.markdown("---")

    if not os.path.exists(PERSIST_DIRECTORY):
        st.warning("Vector database not found. Please ingest a video first.")
        return

    # init chain & vector_store into session_state
    if "chain" not in st.session_state:
        st.session_state.chain, st.session_state.vector_store = load_conversation_chain()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.chain:
        st.stop()

    # suggested prompts
    prompt_buttons = {
        "Iron-Rich Foods": "What iron-rich foods are good for babies?",
        "Protein-Rich Foods": "What protein-rich foods are good for babies?",
        "Brain-Boosting Meals": "What are some brain-boosting foods for babies?",
        "Recovery from Illness": "What foods are recommended for a baby recovering from illness?",
        "Calcium-Rich Options": "What are some good sources of calcium for babies?",
        "Sweet Food Ideas": "What sweet foods can I give my baby?",
        "General Tips": "Give me some general tips for introducing solids.",
        "Recipe of the Day": (
            "Provide one baby-safe recipe for today. "
            "Include ingredients, step-by-step preparation instructions, and safety notes. "
            "DO NOT use the word 'metadata' in your reply."
        )
    }

    btn_cols = st.columns(4)
    final_prompt = None
    for i, (label, question) in enumerate(prompt_buttons.items()):
        if btn_cols[i % 4].button(label):
            final_prompt = question

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt_text = st.chat_input("Ask about ingredients, steps, or cooking times...")
    st.markdown("<p style='text-align: center; color: gray;'>— OR —</p>", unsafe_allow_html=True)
    audio_info = mic_recorder(start_prompt="Record your question 🎤", stop_prompt="Click to Stop ⏹️", key='recorder')

    if prompt_text:
        final_prompt = prompt_text
    elif audio_info:
        audio_bytes = audio_info["bytes"]
        with st.spinner("Transcribing your voice..."):
            final_prompt = transcribe_audio(audio_bytes)
            st.info(f"Recognized text: \"{final_prompt}\"")

    if final_prompt:
        st.session_state.messages.append({"role": "user", "content": final_prompt})
        with st.chat_message("user"):
            st.markdown(final_prompt)

        with st.spinner("Thinking..."):
            try:
                if final_prompt == prompt_buttons["Recipe of the Day"]:
                    llm = LangchainOpenAI(temperature=0)
                    answer = llm(prompt_buttons["Recipe of the Day"])
                else:
                    # use chain.invoke for the conversational retrieval chain
                    response = st.session_state.chain.invoke({"question": final_prompt})
                    answer = response.get("answer", "Sorry, I couldn't find a good answer.")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                with st.chat_message("assistant"):
                    st.markdown(answer)
                generate_and_play_audio(answer)
            except Exception as e:
                error_msg = f"Sorry, an error occurred: {e}"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(error_msg)


if __name__ == "__main__":
    main()
