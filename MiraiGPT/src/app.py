import streamlit as st
import os
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
import tempfile
import warnings

# Turn off all warnings
warnings.filterwarnings("ignore")

os.environ["GOOGLE_API_KEY"] = "AIzaSyC6A1MJR-kk-KetpF3Llqna_GE4hulhwMU"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "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"

PERSIST_DIRECTORY = './MiraiGPT/chroma_db'
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.6)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma(
    persist_directory=PERSIST_DIRECTORY,
    embedding_function=embedding_model
)

st.markdown("""
<style>
.stApp {
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.title("Tra cứu tài liệu với MiraiGPT")

st.subheader("Thêm tài liệu PDF mới")
uploaded_file = st.file_uploader("Chọn file PDF", type=["pdf"])

# Function to process and add new PDF document to vectorstore
def add_pdf_to_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    vectorstore.add_documents(splits)
    st.success("Tài liệu đã được thêm vào vectorstore thành công!")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        add_pdf_to_vectorstore(tmp_file.name)

st.subheader("Hỏi đáp cùng MiraiGPT")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


retriever = vectorstore.as_retriever()
prompt_template = """
    Bạn là trợ lý AI được yêu cầu trả lời câu hỏi dựa trên tài liệu sau nếu có thông tin:
    {context}

    Nếu không có thông tin liên quan trong tài liệu, hãy trả lời câu hỏi dựa trên kiến thức của bạn và không cần đề cập đến tài liệu.
    Câu hỏi: {question}
    """

# prompt_template= """You are an assistant that answers questions based on the following retrieved context.
#         If the answer can be found within the context, provide that answer.
#         if the context does not contain sufficient information, use your general knowledge to answer
#         the question. Always aim to provide a concise and accurate answer. No yapping\n\n
#         *Question*: {question}\n
#         *Context*: {context}\n\n
#         *Answer*:"""



def create_prompt(context, question):
    return prompt_template.format(context=context, question=question)


# Chat interface for Q&A
if "messages" not in st.session_state:
    st.session_state.messages = []

# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Khung chat để người dùng nhập câu hỏi
user_prompt = st.text_input("Nhập câu hỏi của bạn:")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})

    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý câu trả lời..."):
            try:
                # Ensure these functions are correctly defined and error-handled
                docs = retriever.invoke(user_prompt)  # Retrieve relevant documents
                context = format_docs(docs)  # Format the documents
                prompt = create_prompt(context, user_prompt)  # Create prompt for LLM
                response = llm.invoke(prompt).content  # Get response from LLM
            except Exception as e:
                response = f"Đã xảy ra lỗi: {str(e)}"

        #Display assistant's response
        st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

        # Scroll to the bottom of the page using JavaScript
        st.markdown("""
                <script>
                var chatBox = window.parent.document.querySelector('section.main');
                chatBox.scrollTop = chatBox.scrollHeight;
                </script>
                """, unsafe_allow_html=True)

user_prompt = ''