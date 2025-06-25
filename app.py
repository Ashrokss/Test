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

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
PDF_FOLDER = r"C:\Users\AshishPal\Downloads\RAG-PDF_TO_CHATBOT\RAG-PDF_TO_CHATBOT\Data\pdf"

EMBEDDED_DATA_PATH = r"C:\Users\AshishPal\Downloads\RAG-PDF_TO_CHATBOT\RAG-PDF_TO_CHATBOT\Data\EmbeddedData\faiss_index"


# Initialize LLM
@st.cache_resource
def initialize_llm():
    try:
        return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
    except Exception as e:
        st.error(f"Error initializing LLM: {e}")
        st.stop()

llm = initialize_llm()

# Define prompt template
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

# Initialize embeddings
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

# Process PDF files and generate embeddings
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

# Load FAISS index
def load_faiss_index(embedded_data_path):
    try:
        return FAISS.load_local(embedded_data_path, initialize_embeddings(), allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return None

# Generate or Load Embeddings
if not os.path.exists(EMBEDDED_DATA_PATH):
    st.write("FAISS index not found. Generating embeddings from the PDF files...")
    embeddings = initialize_embeddings()
    vectors = process_pdfs(PDF_FOLDER, embeddings, EMBEDDED_DATA_PATH)
    if vectors:
        st.success("Embeddings generated and saved successfully!")
else:
    # st.write("Loading FAISS index from embedded data...")
    vectors = load_faiss_index(EMBEDDED_DATA_PATH)

# Question and Answer Section
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


# import streamlit as st
# import os
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_community.vectorstores import FAISS
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq
# from langchain.chains import RetrievalQA
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# # Paths
# PDF_FOLDER = r"D:\RAG\Data\pdf"
# EMBEDDED_DATA_PATH = r"D:\RAG\Data\EmbeddedData\faiss_index"

# # Initialize LLM
# @st.cache_resource
# def initialize_llm():
#     try:
#         return ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Llama3-8b-8192")
#     except Exception as e:
#         st.error(f"Error initializing LLM: {e}")
#         st.stop()

# llm = initialize_llm()

# # Define multiple prompt templates
# prompts = [
#     ChatPromptTemplate.from_template(
#         """
#         Based on the provided context, explain the working of the tool.

#         <context>
#         {context}
#         <context>
#         Question: {input}
#         """
#     ),
#     ChatPromptTemplate.from_template(
#         """
#         Based on the provided context, describe the name and working of the tool in detail.

#         <context>
#         {context}
#         <context>
#         Question: {input}
#         """
#     ),
#     ChatPromptTemplate.from_template(
#         """
#         Based on the provided context, provide a detailed guide on how to repair the tool.

#         <context>
#         {context}
#         <context>
#         Question: {input}
#         """
#     )
# ]

# # Initialize embeddings
# @st.cache_resource
# def initialize_embeddings():
#     try:
#         return GoogleGenerativeAIEmbeddings(
#             model="models/embedding-001",
#             google_api_key=GOOGLE_API_KEY,
#         )
#     except Exception as e:
#         st.error(f"Error initializing embeddings: {e}")
#         st.stop()

# # Process PDF files and generate embeddings
# def process_pdfs(pdf_folder, embeddings, embedded_data_path):
#     try:
#         documents = []
#         for file_name in os.listdir(pdf_folder):
#             if file_name.endswith(".pdf"):
#                 pdf_path = os.path.join(pdf_folder, file_name)
#                 loader = PyPDFLoader(pdf_path)
#                 documents.extend(loader.load())

#         if not documents:
#             st.error("No valid documents found in the PDF folder.")
#             return None

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#         chunks = text_splitter.split_documents(documents)
#         st.write(f"Generated {len(chunks)} text chunks from PDFs.")

#         vectors = FAISS.from_documents(chunks, embeddings)
#         vectors.save_local(embedded_data_path)
#         return vectors
#     except Exception as e:
#         st.error(f"Error processing PDFs: {e}")
#         return None

# # Load FAISS index
# def load_faiss_index(embedded_data_path):
#     try:
#         return FAISS.load_local(embedded_data_path, initialize_embeddings(), allow_dangerous_deserialization=True)
#     except Exception as e:
#         st.error(f"Error loading FAISS index: {e}")
#         return None

# # Generate or Load Embeddings
# if not os.path.exists(EMBEDDED_DATA_PATH):
#     st.write("FAISS index not found. Generating embeddings from the PDF files...")
#     embeddings = initialize_embeddings()
#     vectors = process_pdfs(PDF_FOLDER, embeddings, EMBEDDED_DATA_PATH)
#     if vectors:
#         st.success("Embeddings generated and saved successfully!")
# else:
#     st.write("Loading FAISS index from embedded data...")
#     vectors = load_faiss_index(EMBEDDED_DATA_PATH)

# # Question and Answer Section
# if vectors:
#     question = st.text_input("Enter your question about tools", placeholder="E.g., 'Explain Tool X'")

#     if question:
#         retriever = vectors.as_retriever()
#         try:
#             responses = []
#             for prompt in prompts:
#                 retrieval_chain = RetrievalQA.from_chain_type(
#                     retriever=retriever,
#                     llm=llm,
#                     prompt_template=prompt,
#                     chain_type="map_reduce"  # Ensure we specify a valid chain type
#                 )
#                 response = retrieval_chain({"input": question})
#                 responses.append(response["output_text"])

#             st.write("### Answers")
#             for i, answer in enumerate(responses):
#                 st.write(f"#### Answer {i + 1}")
#                 st.success(answer)

#                 with st.expander("Relevant Context"):
#                     for doc in response.get("context", []):
#                         st.write(doc.page_content)
#                         st.write("---")
#         except Exception as e:
#             st.error(f"Error processing question: {e}")
# else:
#     st.error("Embeddings are not available. Please check the PDF folder or try regenerating the embeddings.")
