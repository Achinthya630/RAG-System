import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings



load_dotenv()


def load_documents(docs_path = "docs"):
  """Load all text files from the docs directory"""
  print(f"Loading documents from {docs_path}...")
    
  
  if not os.path.exists(docs_path):
    raise FileNotFoundError(f"The directory {docs_path} does not exist")

  loader = DirectoryLoader(
    path=docs_path,
    glob = "*.txt",
    loader_cls=TextLoader,
    loader_kwargs={
        "encoding": "utf-8",
        "autodetect_encoding": True
    }
  )

  documents = loader.load()

  if len(documents) == 0:
    raise FileNotFoundError(f"No files found in {docs_path}.")
  
  for i, doc in enumerate(documents[:2]):
    print(f"\nDocument {i+1}:")
    print(f"  Source: {doc.metadata['source']}")
    print(f"  Content length: {len(doc.page_content)} characters")
    print(f"  Content preview: {doc.page_content[:100]}...")
    print(f"  metadata: {doc.metadata}")

  return documents







def split_documents(documents, chunk_size=1000, chunk_overlap=150):
    """Split documents into semantic chunks for RAG"""
    print("Splitting documents into chunks...")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)

    print(f"Total chunks created: {len(chunks)}")

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"Source: {chunk.metadata.get('source', 'unknown')}")
        print(f"Length: {len(chunk.page_content)} characters")
        print(f"Content:")
        print(chunk.page_content)
        print("-" * 50)

    
    if len(chunks) > 5:
      print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks



# cat = [23,45,78]
# kitten = [24,45,77]

# car = [12,64,99]


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    """Create and persist ChromaDB vector store using Gemini embeddings"""
    print("Creating embeddings and storing in ChromaDB...")

    # embedding_model = GoogleGenerativeAIEmbeddings(
    #     # model="models/text-embedding-001"
    #     model="gemini-embedding-001"

    # )

    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


    print("--- Creating vector store ---")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )
    print("--- Finished creating vector store ---")

    print(f"Vector store created and saved to {persist_directory}")
    return vectorstore



def main():
  
  persistent_directory = "db/chroma_db"
    
  if os.path.exists(persistent_directory):
    print("Vector store already exists. No need to re-process documents.")
    
    # embedding_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    # vectorstore = Chroma(
    #     persist_directory=persistent_directory,
    #     embedding_function=embedding_model, 
    #     collection_metadata={"hnsw:space": "cosine"}
    # )

    embedding_model = HuggingFaceEmbeddings(
      model_name="sentence-transformers/all-MiniLM-L6-v2",
      model_kwargs={"device": "cpu"}
    )
    vectorstore = Chroma(
      persist_directory=persistent_directory,
      embedding_function=embedding_model,
      collection_metadata={"hnsw:space": "cosine"}
    )

    print(f"Loaded existing vector store with {vectorstore._collection.count()} documents")
    return vectorstore
    
  print("Persistent directory does not exist. Initializing vector store...\n")

  documents = load_documents(docs_path = "docs")
  chunks = split_documents(documents)
  
  vectorstore = create_vector_store(chunks, persistent_directory)
    
  print("\nIngestion complete! Your documents are now ready for RAG queries.")
  return vectorstore


if __name__ == "__main__":
  main()

  # excel file rag system 