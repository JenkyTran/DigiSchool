import os
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

# Thiết lập API Key nếu chưa có
if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = "AIzaSyC6A1MJR-kk-KetpF3Llqna_GE4hulhwMU"

# Khởi tạo LLM từ Google Generative AI
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.9)

# Tiêu đề của ứng dụng
st.title("Demo MiraiGPT v1 - Hội thoại")

# Khởi tạo session_state để lưu lịch sử các tin nhắn nếu chưa có
if "messages" not in st.session_state:
    st.session_state.messages = []


# Hàm để in toàn bộ log trò chuyện
def print_chat_log():
    st.write("### Lịch sử trò chuyện:")
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.write(f"**Người dùng**: {message['content']}")
        elif message["role"] == "assistant":
            st.write(f"**Trợ lý**: {message['content']}")
    st.write("---")  # Đường kẻ để tách log và giao diện hiện tại


# Hiển thị lịch sử hội thoại
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Khung chat để người dùng nhập câu hỏi
if prompt := st.chat_input("Nhập câu hỏi của bạn:"):
    # Kiểm tra nếu người dùng gửi mã #not2024
    if prompt.strip() == "#not2024":
        print_chat_log()  # Gọi hàm in log khi người dùng gửi #not2024
    else:
        # Thêm câu hỏi của người dùng vào lịch sử
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Hiển thị câu hỏi của người dùng trong khung chat
        with st.chat_message("user"):
            st.markdown(prompt)

        # Lấy phản hồi từ LLM
        with st.chat_message("assistant"):
            response = llm.invoke(prompt)
            assistant_message = response.content

            # Hiển thị phản hồi của trợ lý trong khung chat
            st.markdown(assistant_message)

            # Lưu phản hồi của trợ lý vào lịch sử hội thoại
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})
