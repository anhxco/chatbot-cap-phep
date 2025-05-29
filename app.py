import streamlit as st
from dotenv import load_dotenv
import os
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

load_dotenv()

st.title("🤖 Chatbot hỗ trợ cấp phép")

user_question = st.text_input("Bạn cần hỏi gì?")

if user_question:
    try:
        db = FAISS.load_local("vectorstore", OpenAIEmbeddings())
        qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            retriever=db.as_retriever()
        )
        response = qa.run(user_question)
        st.write("📄 Trả lời:")
        st.write(response)
    except Exception as e:
        st.error(f"Lỗi: {e}")
