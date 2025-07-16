from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load the saved FAISS index
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Build a retriever
retriever = db.as_retriever(search_kwargs={"k": 4, "fetch_k": 12})

# Chat model (GPT-4o mini)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Retrieval-augmented QA chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)

if __name__ == "__main__":
    question = "What is the total rent I would pay?"
    result = qa.invoke(question)
    print(result["result"])
    print("\nSources:")
    for doc in result["source_documents"]:
        print(" -", doc.metadata.get("source", "?"))
