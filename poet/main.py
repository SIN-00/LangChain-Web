from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI

llm = OpenAI()
result = llm.invoke("hi!")
print(result)

# from langchain_openai import ChatOpenAI 

# chat_model = ChatOpenAI()
# result = chat_model.invoke("hi!")
# print(result.content)