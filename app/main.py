import streamlit as st
import os
import tempfile
import base64
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="RAG PDF + Image Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG PDF + Image Chatbot")
st.markdown("Upload a PDF and/or an image, then ask questions!")

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

def image_to_base64(image_file):
    return base64.b64encode(image_file.read()).decode("utf-8")

def ingest_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="/tmp/db_streamlit"
    )
    return vectorstore

def get_chain(vectorstore):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 6}
    )
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1024
    )
    prompt = PromptTemplate.from_template(
        "You are a helpful assistant.\n"
        "Use the context below to answer the question.\n"
        "Quote or paraphrase directly from the context.\n"
        "Only say not found if context has no related info.\n\n"
        "Context: {context}\n"
        "Question: {question}\n\n"
        "Answer:"
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

def ask_with_image(question, image_base64, image_type):
    from groq import Groq
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{image_type};base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            }
        ],
        max_tokens=1024
    )
    return response.choices[0].message.content
    return message.content[0].text

    prompt = f"Look at this image and answer: {question}"
    response = llm.invoke(prompt)
    return response.content

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "image_data" not in st.session_state:
    st.session_state.image_data = None
if "image_type" not in st.session_state:
    st.session_state.image_type = None

with st.sidebar:
    st.header("📁 Upload Files")

    st.subheader("📄 PDF Upload")
    uploaded_pdf = st.file_uploader(
        "Choose a PDF file",
        type="pdf"
    )
    if uploaded_pdf is not None:
        if st.button("Process PDF"):
            with st.spinner("Processing PDF..."):
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_pdf.read())
                    tmp_path = tmp_file.name
                vectorstore = ingest_pdf(tmp_path)
                chain, retriever = get_chain(vectorstore)
                st.session_state.vectorstore = vectorstore
                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.session_state.messages = []
                os.unlink(tmp_path)
            st.success("PDF ready!")

    st.subheader("🖼️ Image Upload")
    uploaded_image = st.file_uploader(
        "Choose an image",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image",
                 width=300)
        uploaded_image.seek(0)
        st.session_state.image_data = base64.b64encode(
            uploaded_image.read()
        ).decode("utf-8")
        st.session_state.image_type = uploaded_image.type
        st.success("Image ready!")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if st.session_state.chain is None and st.session_state.image_data is None:
    st.info("Upload a PDF or an image to start chatting!")
else:
    mode = ""
    if st.session_state.chain and st.session_state.image_data:
        mode = "PDF + Image"
    elif st.session_state.chain:
        mode = "PDF only"
    else:
        mode = "Image only"
    st.caption(f"Mode: {mode}")

    if question := st.chat_input("Ask anything..."):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = ""

                if st.session_state.chain:
                    pdf_answer = st.session_state.chain.invoke(question)
                    answer += pdf_answer

                if st.session_state.image_data:
                    image_answer = ask_with_image(
                        question,
                        st.session_state.image_data,
                        st.session_state.image_type
                    )
                    if st.session_state.chain:
                        answer += "\n\n🖼️ **From image:** " + image_answer
                    else:
                        answer = image_answer

                if st.session_state.retriever:
                    docs = st.session_state.retriever.invoke(question)
                    sources = list(set([
                        f"Page {doc.metadata.get('page', '?')+1}"
                        for doc in docs
                    ]))
                    answer += f"\n\n📌 *Sources: {', '.join(sources)}*"

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )