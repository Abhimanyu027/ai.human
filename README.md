# 🌐 Website AI Chatbot (RAG-based)

A **Website-based AI Chatbot** built using **Streamlit, LangChain, OpenAI, and FAISS**.  
The app allows users to **index any public website** and ask questions that are answered **strictly from the website content** using a **Retrieval-Augmented Generation (RAG)** approach.

---

## 🚀 Features

- 🔗 Crawl and extract text from any website URL
- ✂️ Smart text chunking using Recursive Character Text Splitter
- 🧠 Semantic search using OpenAI embeddings
- ⚡ Fast similarity search with FAISS
- 💬 Conversational Q&A with memory support
- 🔐 Secure OpenAI API key input (not stored)
- 🎨 Simple and interactive Streamlit UI

---

## 🧱 Tech Stack

- **Frontend / UI**: Streamlit  
- **Backend**: Python  
- **Web Scraping**: Requests, BeautifulSoup  
- **LLM**: OpenAI (via LangChain)  
- **Embeddings**: OpenAI Embeddings  
- **Vector Database**: FAISS  
- **Framework**: LangChain  

---

## 🏗️ Architecture (RAG Flow)

1. Website URL is scraped and cleaned
2. Text is split into overlapping chunks
3. Each chunk is converted into vector embeddings
4. Embeddings are stored in a FAISS vector database
5. User questions are matched with relevant chunks
6. LLM generates answers using retrieved context only

---

## 📁 Project Structure


---

## 🛠️ Installation (Local Setup)

###
1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
2️⃣ Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
3️⃣ Install dependencies
pip install -r requirements.txt
▶️ Run the Application
streamlit run human.py


The app will be available at:

http://localhost:8501

Live link:
https://aihuman-nt9xf5rawpdmsu94cvw5ll.streamlit.app/
