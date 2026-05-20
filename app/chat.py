from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

def load_qa_chain():
    print("Loading vector database...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory="./db",
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )

    print("Setting up Groq AI...")
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=2048
    )

    prompt = PromptTemplate.from_template(
        "You are a helpful assistant analyzing a document.\n"
        "Use the context below to answer the question.\n"
        "Quote or paraphrase directly from the context.\n"
        "If the context contains partial information,\n"
        "share what is available.\n"
        "Only say not found if the context has absolutely\n"
        "no related information.\n\n"
        "Context: {context}\n"
        "Question: {question}\n\n"
        "Answer based on the document:"
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs,
         "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def ask_question(chain, retriever, question):
    print(f"\nQuestion: {question}")
    print("Thinking...")
    answer = chain.invoke(question)
    print(f"\nAnswer: {answer}")
    docs = retriever.invoke(question)
    print("\n--- Sources used ---")
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "?")
        print(f"Source {i+1}: Page {page+1}")

if __name__ == "__main__":
    chain, retriever = load_qa_chain()
    print("\nReady! Ask me anything about your PDF.")
    print("Type exit to quit\n")
    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            break
        ask_question(chain, retriever, question)