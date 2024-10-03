import streamlit as st
import os
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Thiết lập biến môi trường
os.environ["GOOGLE_API_KEY"] = "AIzaSyC6A1MJR-kk-KetpF3Llqna_GE4hulhwMU"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

from langchain_google_genai import ChatGoogleGenerativeAI

# Khởi tạo LLM từ Google Generative AI
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.9)

# Tiêu đề của ứng dụng
st.title("Tra cứu tài liệu với MiraiGPT")

# Tạo một khung nhập văn bản cho người dùng
user_input = st.text_input("Nhập câu tài liệu của bạn:", "")

# Nếu văn bản đầu vào không rỗng
if user_input:
    # Tạo document từ đầu vào của người dùng
    docs = [Document(page_content=user_input)]

    # Tạo mô hình embedding từ sentence-transformers
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Tách văn bản thành các phần nhỏ
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Kiểm tra nếu có dữ liệu sau khi chia nhỏ
    if splits:
        # Tạo vectorstore từ các đoạn văn bản đã chia
        vectorstore = Chroma.from_documents(documents=splits, embedding=embedding_model)

        # Retrieve and generate using the relevant snippets
        retriever = vectorstore.as_retriever()
        prompt_template = """
            You are an AI assistant asked to answer the following document-based question if information is available:
            {context}

            If there is no relevant information in the document, answer the question based on your knowledge. 
            Question: {question}
            """

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        # Nhập câu hỏi của người dùng
        if user_prompt := st.chat_input("Nhập câu hỏi của bạn:"):
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                response = rag_chain.invoke(user_prompt)

                # Hiển thị phản hồi của trợ lý trong khung chat
                st.markdown(response)
    else:
        st.error("Không thể tách văn bản thành các phần nhỏ.")
else:
    st.warning("Vui lòng nhập câu tài liệu của bạn.")
