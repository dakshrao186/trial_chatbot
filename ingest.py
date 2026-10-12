import os
import pandas as pd
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
load_dotenv()
documents = []
# Load TXT and PDF files
DATA_FOLDER = "data"
if os.path.exists(DATA_FOLDER):
    for file in os.listdir(DATA_FOLDER):
        path = os.path.join(DATA_FOLDER, file)
        if file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
            documents.extend(loader.load())
            print(f"Loaded TXT: {file}")
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
            documents.extend(loader.load())
            print(f"Loaded PDF: {file}")
# Load CSV files
DATASET_FOLDER = "dataset"
if os.path.exists(DATASET_FOLDER):
    for root, dirs, files in os.walk(DATASET_FOLDER):
        for file in files:
            if file.endswith(".csv"):
                csv_path = os.path.join(root, file)
                print(f"Loading CSV: {file}")
                df = pd.read_csv(csv_path)
                for index, row in df.iterrows():
                    text = ""
                    for column in df.columns:
                        text += f"{column}: {row[column]}\n"
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file,
                                "row": index
                            }
                        )
                    )
print(f"\nLoaded {len(documents)} documents.")
# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
docs = text_splitter.split_documents(documents)
print(f"Created {len(docs)} chunks.")
# Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("Embedding model loaded.")
# Create FAISS database
db = FAISS.from_documents(
    docs,
    embeddings
)
db.save_local("vector_db")
print("\nVector database created successfully!")