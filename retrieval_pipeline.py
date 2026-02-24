from langchain_chroma import Chroma
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"

# embedding_model = GoogleGenerativeAIEmbeddings(model = "models/text-embedding-004")

# ---------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

db = Chroma(
  persist_directory = persistent_directory,
  embedding_function = embedding_model,
  collection_metadata = {"hnsw:space":"cosine"}
)

# query = "Who had given the budget speech in 2020?"
# query = "what is said about carbon neutral economy?"
# query = "what is said about the development of trains?"
# query = "what difference is made in the tax for futures and options in the year 2026, and how was it before?"
query = "Explain the kisan credit card?"




retriever = db.as_retriever(search_kwargs = {"k":5})

relevant_docs = retriever.invoke(query)

print(f"User query: {query}")
print("--------Context--------")
for i, doc in enumerate (relevant_docs,1):
  print(f"Document {i}: \n{doc.page_content}\n")


combined_input = f"""Based on the following documents, please answer the following question: {query}.

Documents:
{chr(10).join([f"-{doc.page_content}" for doc in relevant_docs])}

Please provide a clear, concise answer using only the information from these documents. Include the from which you are getting the answer as well. If you can't find the relevant information from the documents, then reply back saying the documents do not contain any information related to the query."""


model = ChatGoogleGenerativeAI(
  model = 'gemini-2.5-flash',
  temperature = 1,
  google_api_key = os.getenv("GEMINI_API_KEY")
)


response = model.invoke(combined_input)

print("\n ANSWER: \n")
print(response.content)
