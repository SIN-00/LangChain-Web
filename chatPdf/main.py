from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

load_dotenv()

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

question = "어떤 용어를 설명중이야?"
llm = ChatOpenAI(temperature=0)
retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), llm=llm
)

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

# QA 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),  # or db.as_retriever()
    return_source_documents=True   # (선택) 출처 문서 포함 여부
)
result = qa_chain.run(question)
print(result)