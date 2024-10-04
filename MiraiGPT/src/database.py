import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH = "./MiraiGPT/bookdb/doc.pdf"
PERSIST_DIRECTORY = './MiraiGPT/chroma_db'


def main():
    # Xóa dữ liệu trong Chroma vectorstore trước khi thêm mới
    clear_chroma_data()

    # Load the documents from the PDF
    documents = load_documents()

    # Split the documents into chunks
    chunks = split_documents(documents)

    # Add the chunks to Chroma vector store and persist the data
    add_to_chroma(chunks)

    print('success')


def clear_chroma_data():
    """Xóa dữ liệu trong thư mục Chroma persist_directory."""
    if os.path.exists(PERSIST_DIRECTORY):
        shutil.rmtree(PERSIST_DIRECTORY)
        print(f"Dữ liệu trong {PERSIST_DIRECTORY} đã được xóa.")
    else:
        print(f"Thư mục {PERSIST_DIRECTORY} không tồn tại, không cần xóa.")


def load_documents():
    # Load the PDF file (single PDF, not a directory of PDFs)
    document_loader = PyPDFLoader(DATA_PATH)
    return document_loader.load()


# Khởi tạo mô hình embedding
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Tạo vector store từ các chunk đã tách
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=PERSIST_DIRECTORY
    )
    # Lưu vectorstore để dùng cho các lần sau
    vectorstore.persist()


if __name__ == "__main__":
    main()
