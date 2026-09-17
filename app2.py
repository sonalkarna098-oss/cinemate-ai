import os
import tempfile
import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CineMate AI",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS (Fixed Visibility & Strict Light Contrast)
# ============================================================

st.markdown("""
<style>
/* GLOBAL BACKGROUND & BASE TEXT */
.stApp {
    background-color: #F7F5F0 !important;
    color: #2D241E !important;
}

.block-container {
    max-width: 950px;
    padding-top: 2.5rem;
    padding-bottom: 5rem;
}

/* FORCE ALL TEXT IN APP TO BE DARK AND VISIBLE */
p, span, div, label, h1, h2, h3, h4, h5, h6 {
    color: #2D241E !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #EFECE6 !important;
    border-right: 1px solid #D1C9BC !important;
}

/* BRANDING & HEADERS */
.brand-title {
    font-size: 1.8rem;
    font-weight: 800;
    color: #4A3B32 !important;
    text-align: center;
    margin-bottom: 0px;
}

.brand-subtitle {
    font-size: 0.85rem;
    color: #6E635B !important;
    text-align: center;
    margin-bottom: 20px;
}

/* CARDS & CONTAINERS */
.status-card {
    padding: 12px 16px;
    border-radius: 10px;
    background-color: #E4DFC3 !important;
    border: 1px solid #C4BD9B !important;
    color: #382E27 !important;
    font-size: 0.88rem;
    font-weight: 600;
    margin-bottom: 15px;
    word-break: break-word;
}

/* CHAT MESSAGES */
[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border-radius: 12px !important;
    border: 1px solid #DDD7CD !important;
    margin-bottom: 14px !important;
    padding: 16px !important;
    box-shadow: 0 2px 6px rgba(0,0,0,0.03) !important;
}

/* FIX SPINNER AND CHAT CONTENT VISIBILITY */
[data-testid="stChatMessage"] * {
    color: #2D241E !important;
}

/* CHAT INPUT FIELD */
[data-testid="stChatInput"] {
    border-radius: 12px !important;
}

[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    border: 1px solid #C5BFB5 !important;
    color: #2D241E !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: #8C827A !important;
}

[data-testid="stChatInput"] textarea:focus {
    border-color: #8C6D58 !important;
    box-shadow: 0 0 0 2px rgba(140, 109, 88, 0.2) !important;
}

/* BUTTONS */
.stButton > button {
    border-radius: 8px !important;
    background-color: #E0DDD5 !important;
    border: 1px solid #B8AEA1 !important;
    color: #4A3B32 !important;
    font-weight: 600 !important;
    width: 100% !important;
}

.stButton > button:hover {
    background-color: #8C6D58 !important;
    color: #FFFFFF !important;
    border-color: #8C6D58 !important;
}

.stButton > button * {
    color: inherit !important;
}

/* FILE UPLOADER FIX */
[data-testid="stFileUploader"] {
    background-color: #F0ECE4 !important;
    border-radius: 10px !important;
    border: 1px dashed #A89F93 !important;
    padding: 10px !important;
}

[data-testid="stFileUploader"] * {
    color: #382E27 !important;
}

[data-testid="stFileUploader"] button {
    background-color: #FFFFFF !important;
    border: 1px solid #C5BFB5 !important;
    color: #382E27 !important;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "file_name" not in st.session_state:
    st.session_state.file_name = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown('<div class="brand-title">☕ CineMate AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Offline Movie Assistant</div>', unsafe_allow_html=True)
    
    st.divider()

    # KNOWLEDGE STATUS
    if st.session_state.vector_db:
        st.markdown(f"""
        <div class="status-card">
            📄 <b>Loaded:</b><br>{st.session_state.file_name}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.divider()

    # PDF UPLOADER AT BOTTOM
    st.markdown("### 📄 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

    # PROCESS PDF
    if uploaded_file and st.session_state.file_name != uploaded_file.name:
        with st.status("Reading PDF and indexing...", expanded=False) as status:
            temp_dir = tempfile.mkdtemp()
            temp_filepath = os.path.join(temp_dir, uploaded_file.name)

            with open(temp_filepath, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_filepath)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )

            st.session_state.vector_db = FAISS.from_documents(chunks, embeddings)
            st.session_state.file_name = uploaded_file.name
            
            status.update(label="Ready", state="complete")
            st.rerun()


# ============================================================
# MAIN APPLICATION BODY
# ============================================================

if st.session_state.vector_db is None:
    st.markdown("<h2 style='text-align: center; color: #4A3B32 !important; margin-top: 40px;'>Welcome to CineMate AI</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #6E635B !important; font-size: 1.05rem;'>Upload a movie PDF using the panel on the left sidebar to start chatting.</p>", unsafe_allow_html=True)

else:
    # CHAT HISTORY DISPLAY
    for message in st.session_state.messages:
        avatar = "🧑‍💻" if message["role"] == "user" else "☕"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

    # CHAT INPUT
    question = st.chat_input("Ask a question about your document...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)

        llm = OllamaLLM(model="llama3.2:1b")
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 5})

        history_text = ""
        for msg in st.session_state.messages[-5:-1]:
            role = "User" if msg["role"] == "user" else "AI"
            history_text += f"{role}: {msg['content']}\n"

        prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant. Answer the user's question using ONLY the provided document context.

DOCUMENT CONTEXT:
{context}

CHAT HISTORY:
{history}

QUESTION:
{question}

ANSWER:
""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        with st.chat_message("assistant", avatar="☕"):
            with st.spinner("Searching document..."):
                retrieved_docs = retriever.invoke(question)
                context_text = format_docs(retrieved_docs)

                chain = prompt | llm | StrOutputParser()
                answer = chain.invoke({
                    "context": context_text,
                    "history": history_text,
                    "question": question
                })

                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})