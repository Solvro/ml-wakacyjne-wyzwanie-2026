#!/usr/bin/env python
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain.agents import create_agent


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
        chunk_size=200,
        chunk_overlap=50,
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

def find(vector_base, subject):
    retriever = vector_base.as_retriever(
        search_kwargs={"k": 10}
        )

    similar = retriever.invoke(subject)
    for doc in similar:
      print(doc.page_content[:300] + "...\n---")

@tool
def search_knowledge_base(query:str) -> str:
    """Searches the knowledge base for information from the loaded PDF documents.
    Use this tool when the user asks about the content of the documents.
    """
    print("DEBUG: search_knowledge_base used")

    retriever = vector_base.as_retriever(
        search_kwargs={"k": 10}
        )

    docs = retriever.invoke(query)

    if not docs:
        return "No matching information in the knowledge base."

    # Wyciągamy samą treść z każdego obiektu w docs i łączymy w jeden tekst
    return "\n\n---\n\n".join([doc.page_content for doc in docs])


tools = [search_knowledge_base]



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

# Create system prompt
SYSTEM_PROMPT="""
Jesteś asystentem użytkownika, który właśnie stał się jednym z ocalałych pasażerów katastrofy Titanica, rozbitym na bezludnej wyspie. Masz za zadanie pomóc mu w czynnościach technicznych i odpowiadania na pytania dotyczące bazy danych jaka została ci udostępniona.
    1. Jeśli użytkownik zapyta cię o coś z bazy, użyj `search_knowledge_base`.
By poprawić humor świeżemu rozbitkowi, każdą wypowiedź zaczynaj słowami `Achoj!` i udawaj pirata. Używaj emoji do wzbogacania odpowiedzi!
"""

agent = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=tools,
    system_prompt=SYSTEM_PROMPT
    )

result1 = agent.invoke({"messages": [{"role": "user", "content": "Przeszukaj bazę i kilkoma zdaniami opisz jej tematykę"}]})
def parse_result(res):
  to_parse = res["messages"][-1]
  return to_parse.content[0]['text']

print(parse_result(result1))

