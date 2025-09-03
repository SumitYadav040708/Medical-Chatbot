from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader

## Uncomment the following files if you're not using pipenv as your virtual environment manager
#from dotenv import load_dotenv, find_dotenv
#load_dotenv(find_dotenv())


# Step 1: Load raw PDF(s)
DATA_PATH="data/"
import os

def load_pdf_files(data):
    documents = []
    for filename in os.listdir(data):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}...")   # 👈 log which PDF is loading
            pdf_path = os.path.join(data, filename)
            loader = PyMuPDFLoader(pdf_path)
            docs = loader.load()
            print(f" -> {len(docs)} pages extracted")   # 👈 log how many pages
            documents.extend(docs)
    return documents


documents = load_pdf_files(data=DATA_PATH)
#print("Length of PDF pages: ", len(documents))


# Step 2: Create Chunks
def create_chunks(extracted_data):
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                 chunk_overlap=50)
 
    text_chunks=text_splitter.split_documents(extracted_data)
    return text_chunks

print(f"Splitting into chunks...")
text_chunks=create_chunks(extracted_data=documents)
print(f" -> {len(text_chunks)} chunks created")
#print("Length of Text Chunks: ", len(text_chunks))

# Step 3: Create Vector Embeddings 

def get_embedding_model():
    embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2") 
    return embedding_model

embedding_model=get_embedding_model()

# Step 4: Store embeddings in FAISS
DB_FAISS_PATH="vectorstore/db_faiss"
print("Creating embeddings (this may take a while)...")
db=FAISS.from_documents(text_chunks,embedding_model)
print(" -> Embeddings stored in FAISS")
db.save_local(DB_FAISS_PATH)
print(f"FAISS database saved at {DB_FAISS_PATH}")