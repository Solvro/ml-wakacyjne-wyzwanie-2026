#!/usr/bin/env python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


pdf_paths = [
        './pdfy/babeczki_kokosowe.pdf',
        './pdfy/kokosanki.pdf',
        './pdfy/lody_kokosowe.pdf',
        './pdfy/kokosanka.pdf',
        './pdfy/likier_kokosowy.pdf'
        ]

def load_pdf(pdf_path):
    loader = PyMuPDFLoader(file_path=pdf_path)
    return loader.load()

def chunk_pdfs(loaded_pdfs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.split_documents(loaded_pdfs)
    return docs

def create_vector_base(docs):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        # persist_directory="/content/drive/MyDrive/Colab Notebooks/ZadanieAgenty/"
        )
    return vectorstore



# model key loading
_ = load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# check if loaded successfully
if not api_key:
    raise ValueError("No key found!")
else:
    print("Key loaded!")

# loading pdfs
pdfs_loaded = []
for pdf_path in pdf_paths:
    pdf_loaded = load_pdf(pdf_path)
    print("Loaded pages:", len(pdf_loaded))
    pdfs_loaded.extend(pdf_loaded)

# chunking pdfs
docs = chunk_pdfs(pdfs_loaded)
print("Loaded chunks:", len(docs))

# get vector store
vector_base = create_vector_base(docs)
