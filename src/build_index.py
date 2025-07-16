from pathlib import Path
from dotenv import load_dotenv
import pickle

chunks = pickle.load(open("artifacts/chunks.pkl", "rb"))

load_dotenv(Path(__file__).resolve().parents[1] / ".env")  

from ingest import chunks
from langchain_openai import OpenAIEmbeddings                 
from langchain_community.vectorstores import FAISS

emb = OpenAIEmbeddings(model="text-embedding-3-small")
db  = FAISS.from_documents(chunks, emb)
db.save_local("faiss_index")                                       
print("Saved FAISS index to ./faiss_index")