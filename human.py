import os
import streamlit as st
import requests
from bs4 import BeautifulSoup

# --- New Imports for OpenAI ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# ===================== CONFIG =====================
VECTOR_DB_PATH = "faiss_index_openai"

# ===================== WEBSITE SCRAPER =====================
def extract_website_text(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Cleanup
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
            
        text = soup.get_text(separator=" ")
        cleaned_text = " ".join(text.split())
        
        if len(cleaned_text) < 200:
            return None
            
        return cleaned_text
    except Exception:
        return None

# ===================== VECTOR DATABASE =====================
def create_vector_store(text, api_key):
    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, # Increased chunk size for OpenAI (it handles more context)
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    # Use OpenAI Embeddings (Cloud-based, no local Torch needed)
    embeddings = OpenAIEmbeddings(openai_api_key="YOUR KEY")
    
    vector_db = FAISS.from_texts(chunks, embeddings)
    vector_db.save_local(VECTOR_DB_PATH)

def load_vector_store(api_key):
    embeddings = OpenAIEmbeddings(openai_api_key="YOUR KEY")
    return FAISS.load_local(
        VECTOR_DB_PATH, 
        embeddings, 
        allow_dangerous_deserialization=True
    )

# ===================== QA CHAIN =====================
def build_qa_chain(vector_db, api_key):
    # Use GPT-3.5-Turbo (Fast, cheap, smart)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0, 
        openai_api_key=api_key
    )

    prompt_template = """
    You are a helpful assistant. Answer the question specifically based on the context provided below.
    If the answer is not in the context, say "I cannot find the answer on this website."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "question"], 
        template=prompt_template
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_db.as_retriever(search_kwargs={"k": 5}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=False
    )

# ===================== STREAMLIT UI =====================
st.set_page_config(page_title="Website AI Chatbot", layout="centered")
st.title("🌐 Website-Based AI Chatbot (OpenAI)")

# --- Sidebar for Secure API Key Entry ---
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    st.caption("Key is not saved permanently.")

if not api_key:
    st.warning("⚠️ Please enter your OpenAI API Key in the sidebar to continue.")
    st.stop()

# --- Main App ---
url = st.text_input("Enter Website URL")

if st.button("Index Website"):
    with st.spinner("Crawling and indexing website..."):
        website_text = extract_website_text(url)
        if not website_text:
            st.error("Invalid or unsupported website.")
        else:
            create_vector_store(website_text, api_key)
            st.success("Website indexed successfully!")

st.divider()

question = st.text_input("Ask a question related to the website")

if question:
    if not os.path.exists(VECTOR_DB_PATH):
        st.warning("Please index a website first.")
    else:
        try:
            vector_db = load_vector_store(api_key)
            qa_chain = build_qa_chain(vector_db, api_key)
            
            response = qa_chain.invoke({"question": question})
            st.write(response["answer"])
            
        except Exception as e:
            st.error(f"Error: {e}")