from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("economic.pdf")
pages = loader.load_and_split()

print(pages[200])