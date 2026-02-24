from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os


from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"

# embedding_model = GoogleGenerativeAIEmbeddings(
#     model="models/text-embedding-004"
# )

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)


db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"}
)

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0.1,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

chat_history = []

def ask_question(user_question: str):
    print(f"\n--- You asked: {user_question} ---")

    if chat_history:
        messages = [
            SystemMessage(
                content=(
                    "Given the chat history, rewrite the new question to be "
                    "standalone and searchable. Return ONLY the rewritten question."
                )
            )
        ] + chat_history + [
            HumanMessage(content=f"New question: {user_question}")
        ]

        result = model.invoke(messages)

        # if isinstance(result.content, list):
        #     answer = "".join([part["text"] if isinstance(part, dict) else str(part) 
        #                 for part in result.content])
        # else:
        #     answer = result.content


        if isinstance(result.content, list):
            parts = []
            for part in result.content:
                if isinstance(part, dict):
                    parts.append(part.get("text") or part.get("content") or "")
                else:
                    parts.append(str(part))
            search_question = "".join(parts).strip()
        else:
            search_question = str(result.content).strip()


        # search_question = result.content.strip()
        print(f"🔎 Searching for: {search_question}")

    else:
        search_question = user_question

    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"\n📄 Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f"  Doc {i}: {preview}...")

    combined_input = f"""
Answer the question ONLY using the provided documents.

The answer must contain two sections - one section should contain the answer only from the documents provided, the other should contain the general answer. if the answer is not there in the provided documents, specify in the first section that the given information isn't enough to answer the question. Also, ensure that the overall answer is somewhat related to the content provided, i.e, related to the budget of India.

User Question:
{user_question}

Retrieved Documents:
{"\n".join([f"- (Source: {doc.metadata.get('source','unknown')}) {doc.page_content}" for doc in docs])}
"""

    messages = [
        SystemMessage(
            content=(
                "You are a helpful assistant that answers questions "
                "based ONLY on the provided documents and conversation history."
            )
        )
    ] + chat_history + [
        HumanMessage(content=combined_input)
    ]

# if using gemini 3 flash, use this :

    response = model.invoke(messages)

    if isinstance(response.content, list):
        answer = "".join([part["text"] if isinstance(part, dict) else str(part) 
                        for part in response.content])
    else:
        answer = response.content


# If using gemini 2.5 flash, use this
    # result = model.invoke(messages)
    # answer = result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print("\n✅ ANSWER:\n", answer)
    return answer

def start_chat():
    print("Ask me questions! Type 'quit' to exit.")

    while True:
        question = input("\nYour question: ")

        if question.lower() == "quit":
            print("Goodbye!")
            break

        ask_question(question)


if __name__ == "__main__":
    start_chat()


