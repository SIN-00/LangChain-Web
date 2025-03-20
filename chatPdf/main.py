from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader("economic.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20, #겹치는 글자정해서 문맥
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(pages)

print(texts[0])