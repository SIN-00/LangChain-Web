# from dotenv import load_dotenv
# load_dotenv()
# llama2 use

from langchain_openai import ChatOpenAI 
import streamlit as st
from langchain_community.llms import CTransformers

# chat_model = ChatOpenAI()
llm = CTransformers(
    model="TheBloke/Llama-2-7B-Chat-GGML",  # ✅ 정확한 모델 ID 사용
    model_file="llama-2-7b-chat.ggmlv3.q2_K.bin"  # ✅ 해당 모델의 파일명 유지
)


st.title("인공지능 시인")
content = st.text_input("시의 주제를 제시해주세요")

if st.button("시 작성 요청하기"):
    with st.spinner("시 작성중...", show_time=True):
        result = llm.invoke("write a poem about : " + content)
        st.write(result.content)
