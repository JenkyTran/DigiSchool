import os
if 'GOOGLE_API_KEY' not in os.environ:
    os.environ['GOOGLE_API_KEY'] = "AIzaSyC6A1MJR-kk-KetpF3Llqna_GE4hulhwMU"

from langchain_google_genai import ChatGoogleGenerativeAI

# Create an instance of the LLM, using the 'gemini-pro' model with a specified creativity level
llm = ChatGoogleGenerativeAI(model='gemini-1.5-flash', temperature=0.9)

# Send a creative prompt to the LLM
# response = llm.invoke('Write a paragraph about Viet Nam')
# print(response.content)



import streamlit as st
# Tiêu đề của ứng dụng
st.title("Chat với LLMs")

# Tạo một khung nhập văn bản cho người dùng
user_input = st.text_input("Nhập câu hỏi của bạn:", "")

# Hiển thị nút gửi
if st.button("Gửi"):
    if user_input:
        # Giả sử get_llm_response là một hàm gọi đến mô-đun LLMs
        response = llm.invoke(user_input)  # Hàm này sẽ trả về câu trả lời từ LLMs

        # Hiển thị câu trả lời từ LLMs
        st.write("Phản hồi từ LLMs:")

        st.write(response.content)
    else:
        st.write("Vui lòng nhập câu hỏi.")