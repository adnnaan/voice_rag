import os
import time
import re
import tempfile
import streamlit as st
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CRITICAL CONFIGURATION: LOAD ENVIRONMENT VARIABLES BEFORE LANGCHAIN INITIALIZES ---
from dotenv import load_dotenv
load_dotenv() 

from langchain_community.document_loaders import PyPDFLoader, RecursiveUrlLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Clean, unified standard Google GenAI packages
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

# Native Google GenAI SDK for Safe Multi-modal Audio Handling
from google import genai
from google.genai import types

# Streamlit Microphone Widget Component
from streamlit_mic_recorder import mic_recorder

# Configuration setup for caching layer
EXPORT_DIR = "saved_chats"
os.makedirs("saved_chats", exist_ok=True)


st.set_page_config(page_title="Voice & Text RAG Chatbot", layout="wide")
st.title("🎙️ Voice-Enabled Multi-Source RAG Chatbot")

if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}

# Session State to decouple widget run loops from pipeline execution strings
if "voice_query_text" not in st.session_state:
    st.session_state.voice_query_text = ""

# --- CACHED RESOURCES WITH EXPLICIT API KEY BINDING ---
@st.cache_resource
def load_embedding_model():
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("🚨 Configuration Error: GOOGLE_API_KEY is missing from your secrets/environment variables.")
        st.stop()
    # Ensure google_api_key is explicitly mapped right here:
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",  # Upgrade to the latest stable embedding model standard
        google_api_key=api_key
    )

@st.cache_resource
def get_native_genai_client():
    api_key = os.getenv("GOOGLE_API_KEY")
    return genai.Client(api_key=api_key)

embeddings_model = load_embedding_model()
native_genai_client = get_native_genai_client()

# Sidebar for Knowledge Base Inputs
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Choose PDF files", type="pdf", accept_multiple_files=True)
    
    st.header("2. Crawl a Website")
    url_input = st.text_input("Paste Website URL:", placeholder="https://example.com")
    
    st.caption("💡 *Tip: If information is deep inside a tab/sub-page, paste that exact full URL to guarantee extraction.*")
    st.write("---")
    process_button = st.button("Build Knowledge Base", type="primary")

# Parallel thread worker to process independent PDFs instantly
def load_single_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    try:
        loader = PyPDFLoader(tmp_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_name"] = f"📄 PDF: {uploaded_file.name}"
        return docs
    finally:
        os.remove(tmp_path)

def process_uploaded_pdfs_parallel(uploaded_files, progress_bar, status_text):
    all_documents = []
    total_files = len(uploaded_files)
    status_text.write(f"🚀 Processing {total_files} PDFs simultaneously via background workers...")
    
    with ThreadPoolExecutor(max_workers=min(4, total_files)) as executor:
        futures = {executor.submit(load_single_pdf, f): f for f in uploaded_files}
        for idx, future in enumerate(as_completed(futures)):
            all_documents.extend(future.result())
            progress_bar.progress(int(((idx + 1) / total_files) * 30))
            
    return all_documents

def clean_html_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for element in soup(["script", "style", "noscript", "header", "footer"]):
        element.extract()
        
    extracted_text = soup.get_text()
    clean_text = re.sub(r"\n\n+", "\n\n", extracted_text).strip()
    
    if len(clean_text) < 150:
        blocks = []
        for tag in soup.find_all(['div', 'section', 'article', 'p', 'li']):
            text = tag.get_text(strip=True)
            if text and len(text) > 20:
                blocks.append(text)
        clean_text = "\n\n".join(blocks)
        
    return clean_text

# Unified Knowledge Base Builder
if process_button:
    if not uploaded_files and not url_input.strip():
        st.sidebar.error("Please upload at least one PDF OR paste a website URL.")
    else:
        sidebar_status = st.sidebar.empty()
        sidebar_progress = st.sidebar.progress(0)
        
        combined_documents = []
        
        if uploaded_files:
            combined_documents.extend(process_uploaded_pdfs_parallel(uploaded_files, sidebar_progress, sidebar_status))
            
        if url_input.strip():
            sidebar_status.write(f"🕸️ Connecting to network domain...")
            sidebar_progress.progress(40)
            
            try:
                web_loader = RecursiveUrlLoader(
                    url=url_input.strip(),
                    max_depth=3,
                    extractor=clean_html_extractor,
                    prevent_outside=True
                )
                
                sidebar_status.write("🕸️ Pulling internal web pages in background streams...")
                sidebar_progress.progress(65)
                
                web_docs = web_loader.load()
                if len(web_docs) > 15:
                    web_docs = web_docs[:15]
                
                for doc in web_docs:
                    actual_url = doc.metadata.get("source", url_input.strip())
                    doc.metadata["source_name"] = f"🕸️ Crawled: {actual_url}"
                combined_documents.extend(web_docs)
                
                sidebar_status.write(f"✅ Ready! Processed {len(web_docs)} pages.")
                sidebar_progress.progress(80)
                
            except Exception as e:
                st.sidebar.error(f"Failed to crawl web link: {e}")

        if combined_documents:
            sidebar_status.write("⚡ Partitioning text blocks...")
            sidebar_progress.progress(85)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
            chunks = text_splitter.split_documents(combined_documents)
            
            sidebar_status.write(f"🤖 Generating cloud vectors for {len(chunks)} pieces...")
            sidebar_progress.progress(90)
            
            vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings_model)
            st.session_state.retriever = vectordb.as_retriever(search_kwargs={"k": 4})
            st.session_state.answer_cache.clear()
            
            sidebar_progress.progress(100)
            sidebar_status.write("🎉 **Knowledge Base Live!**")
            
            time.sleep(1.2)
            sidebar_status.empty()
            sidebar_progress.empty()

def get_filename_slug(question_text):
    slug = re.sub(r'[^a-zA-Z0-9_\- ]', '', question_text)[:40].strip().replace(" ", "_").lower()
    return slug if slug else "cached_query"

def save_chat_to_txt(question_text, answer_text, sources_list):
    clean_filename = get_filename_slug(question_text)
    filename = f"cached_{clean_filename}.txt"
    file_path = os.path.join(EXPORT_DIR, filename)
    content = f"QUESTION:\n{question_text}\n--------------------------------------------------\nANSWER:\n{answer_text}\n--------------------------------------------------\nSOURCES USED:\n"
    for src in sources_list:
        content += f"- {src}\n"
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

def check_txt_folder_cache(question_text):
    clean_filename = get_filename_slug(question_text)
    filename = f"cached_{clean_filename}.txt"
    file_path = os.path.join(EXPORT_DIR, filename)
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        try:
            parts = raw_text.split("--------------------------------------------------")
            answer_part = parts[1].replace("ANSWER:\n", "").strip()
            sources_part = parts[2].replace("SOURCES USED:\n", "").strip().split("\n")
            sources_clean = [s.replace("- ", "").strip() for s in sources_part if s.strip()]
            return {"answer": answer_part, "sources": sources_clean}
        except:
            return None
    return None

# --- SWITCHED TO STABLE, DIRECT DEVELOPER API-KEY LLM ROUTE ---
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash", # Highly stable, fast LTS model for LangChain standard integrations
    temperature=0,
    max_output_tokens=1000,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# -----------------------------------------------------------------------------
# VOICE INTERCEPTION & AUDIO TRANSCRIPTION INTERFACE
# -----------------------------------------------------------------------------
st.write("### 🗣️ Ask with Voice")
audio_output = mic_recorder(
    start_prompt="Click to Speak 🎤",
    stop_prompt="Stop Recording 🛑",
    key="mic_recorder_widget"
)

if audio_output and isinstance(audio_output, dict):
    audio_bytes = audio_output.get("bytes", b"")
    if audio_bytes and st.session_state.voice_query_text == "":
        try:
            with st.spinner("🎙️ Transcribing speech via Gemini..."):
                audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
                
                transcription = native_genai_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=[
                        "Transcribe this audio recording exactly. Do not add any intro comments, greeting text, or conversational pleasantries. Output only the pure text transcription.",
                        audio_part
                    ]
                )
                
                st.session_state.voice_query_text = transcription.text.strip()
        except Exception as audio_err:
            st.error(f"Failed to process voice input: {audio_err}")

st.write("---")

# -----------------------------------------------------------------------------
# UNIFIED TEXT PROCESSING INTERFACE
# -----------------------------------------------------------------------------
default_input_text = st.session_state.voice_query_text
question = st.text_input(
    "Ask a question about your uploaded PDFs or referenced website:", 
    value=default_input_text
)

if question:
    st.session_state.voice_query_text = ""
    
    try:
        cache_key = question.strip().lower()
        start_time = time.perf_counter()
        FALLBACK_ERROR = "I could not find that information in the uploaded context."

        if cache_key in st.session_state.answer_cache and st.session_state.answer_cache[cache_key]["answer"].strip() != FALLBACK_ERROR:
            cached_data = st.session_state.answer_cache[cache_key]
            st.subheader("Answer")
            st.write(cached_data["answer"])
            st.caption(f"⏱️ **Response Time (RAM Cache):** {time.perf_counter() - start_time:.4f} seconds")
            st.subheader("Sources Used")
            for src in cached_data["sources"]: st.markdown(src)
            save_chat_to_txt(question, cached_data["answer"], cached_data["sources"])
            
        else:
            file_cached_data = check_txt_folder_cache(question)
            if file_cached_data and file_cached_data["answer"].strip() != FALLBACK_ERROR:
                st.subheader("Answer")
                st.write(file_cached_data["answer"])
                st.caption(f"⏱️ **Response Time (Loaded from Local .txt File):** {time.perf_counter() - start_time:.4f} seconds")
                st.subheader("Sources Used")
                for src in file_cached_data["sources"]: st.markdown(src)
                st.session_state.answer_cache[cache_key] = file_cached_data
                
            else:
                if "retriever" not in st.session_state:
                    st.warning("⚠️ Please build your Knowledge Base via the sidebar first.")
                else:
                    retrieved_docs = st.session_state.retriever.invoke(question)
                    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

                    prompt = f"""Answer the question concisely using only the provided context. 
Finish sentences completely. Do not return markdown code blocks or JSON.

If missing from context, reply exactly with: {FALLBACK_ERROR}

Context:
{context}

Question:
{question}"""
                    
                    st.subheader("Answer")
                    
                    # Clean token-streaming configuration via generic output parser
                    chain = llm | StrOutputParser()
                    full_response = st.write_stream(chain.stream(prompt))
                    
                    elapsed_time = time.perf_counter() - start_time
                    st.caption(f"⏱️ **Live Pipeline Execution Time:** {elapsed_time:.2f} seconds")

                    st.subheader("Sources Used")
                    sources_used = set()
                    for doc in retrieved_docs:
                        src_label = doc.metadata.get("source_name", "Unknown Source")
                        if "page" in doc.metadata and "📄 PDF" in src_label:
                            src_label += f" (Page {doc.metadata['page'] + 1})"
                        sources_used.add(src_label)
                    
                    for source in sources_used: st.markdown(source)

                    sources_list = list(sources_used)
                    st.session_state.answer_cache[cache_key] = {
                        "answer": full_response,
                        "sources": sources_list
                    }
                    save_chat_to_txt(question, full_response, sources_list)

    except Exception as e:
        st.error(f"An error occurred: {e}")