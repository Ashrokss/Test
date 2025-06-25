1. Import Libraries
The code starts by importing necessary libraries and modules for loading documents, generating embeddings, and creating retrieval chains.

python
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
2. Load Environment Variables
Environment variables are loaded to access API keys securely.

python
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
3. Define Paths
The paths for the PDF folder and the embedded data are defined.

python
PDF_FOLDER = r"D:\RAG\Data\pdf"
EMBEDDED_DATA_PATH = r"D:\RAG\Data\EmbeddedData\faiss_index"
4. Initialize LLM (Large Language Model)
The initialize_llm function initializes the LLM using the provided API key. The st.cache_resource decorator caches the result to avoid re-initializing the LLM multiple times.

python
@st.cache_resource
def initialize_llm():
    try:
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.stop()

llm = initialize_llm()
5. Define Prompt Template
A prompt template is defined using ChatPromptTemplate to structure the questions and context for the LLM.

python
prompt = ChatPromptTemplate.from_template(
    """
    Based on the provided context, extract the following details about the tool:
    - Name
    - Working
    - How to repair it

    <context>
    {context}
    <context>
    Question: {input}
    """
)
6. Initialize Embeddings
The initialize_embeddings function initializes the embeddings using the Google API key.

python
@st.cache_resource
def initialize_embeddings():
    try:
        return GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=GOOGLE_API_KEY,
        )
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        st.stop()
7. Process PDFs and Generate Embeddings
The process_pdfs function processes PDF files, splits them into chunks, and generates embeddings using FAISS. It also saves the embeddings locally.

python
def process_pdfs(pdf_folder, embeddings, embedded_data_path):
    try:
        documents = []
        for file_name in os.listdir(pdf_folder):
            if file_name.endswith(".pdf"):
                pdf_path = os.path.join(pdf_folder, file_name)
                loader = PyPDFLoader(pdf_path)
                documents.extend(loader.load())

        if not documents:
            st.error("No valid documents found in the PDF folder.")
            return None

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        st.write(f"Generated {len(chunks)} text chunks from PDFs.")

        vectors = FAISS.from_documents(chunks, embeddings)
        vectors.save_local(embedded_data_path)
        return vectors
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        return None
8. Load FAISS Index
The load_faiss_index function loads the FAISS index from the local path.

python
def load_faiss_index(embedded_data_path):
    try:
        return FAISS.load_local(embedded_data_path, initialize_embeddings(), allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None
9. Generate or Load Embeddings
The code checks if the FAISS index exists. If not, it generates embeddings from the PDF files. Otherwise, it loads the FAISS index from the embedded data.

python
if not os.path.exists(EMBEDDED_DATA_PATH):
    st.write("FAISS index not found. Generating embeddings from the PDF files...")
    embeddings = initialize_embeddings()
    vectors = process_pdfs(PDF_FOLDER, embeddings, EMBEDDED_DATA_PATH)
    if vectors:
        st.success("Embeddings generated and saved successfully!")
else:
    st.write("Loading FAISS index from embedded data...")
    vectors = load_faiss_index(EMBEDDED_DATA_PATH)
10. Question and Answer Section
If embeddings are available, the code accepts a user question, processes it using the retrieval chain, and displays the answer and relevant context.

python
if vectors:
    question = st.text_input("Enter your question about tools", placeholder="E.g., 'Explain Tool X'")

    if question:
        retriever = vectors.as_retriever()
        try:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            with st.spinner("Processing your question..."):
                response = retrieval_chain.invoke({"input": question})

            st.write("### Answer")
            st.success(response["answer"])

            with st.expander("Relevant Context"):
                for doc in response.get("context", []):
                    st.write(doc.page_content)
                    st.write("---")
        except Exception as e:
            st.error(f"Error processing question: {e}")
else:
    st.error("Embeddings are not available. Please check the PDF folder or try regenerating the embeddings.")
Summary
This code integrates several components to create a retrieval-augmented question-answering system using Streamlit. It loads PDF documents, processes them to generate embeddings, and uses a language model to answer questions based on the retrieved documents. The code handles initialization, embedding generation, and user interaction efficiently.

If you have any further questions or need more details on any part of the code, feel free to ask!