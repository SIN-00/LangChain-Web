from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

#Loader
loader = PyPDFLoader("economic.pdf")
pages = loader.load_and_split()

#Split
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20, #겹치는 글자정해서 문맥
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(pages)

#Embedding

embeddings_model = OpenAIEmbeddings()
# load it into Chroma
db = Chroma.from_documents(texts, embeddings_model)
