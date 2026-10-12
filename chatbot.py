import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True
)
retriever = db.as_retriever(
    search_kwargs={"k":3}
)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

def ask(question):
    docs = retriever.invoke(question)

    if not docs:
        return "Sorry, this question is outside the football knowledge base."

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
You are a football assistant.

Answer ONLY using the context below.
If the answer is not found in the context, reply exactly:
"Sorry, this question is outside the football knowledge base."

Context:
{context}

Question:
{question}
"""

    return llm.invoke(prompt).content