import os
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

DATA_DIR = "documents"

def load_documents():
    docs = []
    for file in os.listdir(DATA_DIR):
        path = os.path.join(DATA_DIR, file)
        if file.endswith(".pdf"):
            docs += PyPDFLoader(path).load()
        elif file.endswith(".docx"):
            docs += Docx2txtLoader(path).load()
        elif file.endswith(".txt"):
            docs += TextLoader(path).load()
    return docs

if __name__ == "__main__":
    documents = load_documents()
    if documents:
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(documents, embeddings, persist_directory="vectorstore")
        db.persist()
        print("✅ Đã xử lý và lưu trữ dữ liệu.")
    else:
        print("⚠️ Không có tài liệu nào.")
