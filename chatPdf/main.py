from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

import streamlit as st

load_dotenv()

#제목
st.title("ChatPDF")
st.write("---")

#파일 업로드
import pandas as pd
from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
st.write("---")
#업로드 되면 동작하는 코드
if uploaded_file is not None:
    
    


#Loader
loader = PyPDFLoader("chatPdf/economic.pdf")
pages = loader.load_and_split()

#Split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=20, #겹치는 글자정해서 문맥
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(pages)

#Embedding

embeddings_model = OpenAIEmbeddings()
# load it into Chroma
db = Chroma.from_documents(texts, embeddings_model)

# chat

question = "몇페이지에서 어떤 용어를 설명중이야?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

# QA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever_from_llm,
    return_source_documents=True  # ✅ 변경
)

response = qa_chain.invoke({"query": question})
print("📌 답변:", response["result"])