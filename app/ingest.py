from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import fitz
import base64
import os

load_dotenv()

def extract_images_from_pdf(pdf_path):
    print("Extracting images from PDF...")
    doc = fitz.open(pdf_path)
    image_descriptions = []
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        max_tokens=500
    )
    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_type = base_image["ext"]
                image_base64 = base64.b64encode(
                    image_bytes
                ).decode("utf-8")
                from groq import Groq
                client = Groq(
                    api_key=os.environ.get("GROQ_API_KEY")
                )
                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/{image_type};base64,{image_base64}"
                                    }
                                },
                                {
                                    "type": "text",
                                    "text": "Describe this image in detail. Include all text, numbers, labels, and visual elements you can see."
                                }
                            ]
                        }
                    ],
                    max_tokens=500
                )
                description = response.choices[0].message.content
                image_descriptions.append({
                    "page": page_num + 1,
                    "description": description,
                    "index": img_index + 1
                })
                print(f"Extracted image {img_index+1} from page {page_num+1}")
            except Exception as e:
                print(f"Could not process image on page {page_num+1}: {e}")
    doc.close()
    return image_descriptions

def ingest_pdf(pdf_path):
    print("Step 1: Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages")

    print("Step 2: Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(pages)
    print(f"Created {len(chunks)} text chunks")

    print("Step 3: Extracting images from PDF...")
    image_descriptions = extract_images_from_pdf(pdf_path)
    print(f"Found {len(image_descriptions)} images")

    print("Step 4: Creating embeddings and storing...")
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    from langchain_core.documents import Document
    for img_data in image_descriptions:
        img_doc = Document(
            page_content=f"Image on page {img_data['page']}: {img_data['description']}",
            metadata={
                "page": img_data['page'] - 1,
                "source": pdf_path,
                "type": "image"
            }
        )
        chunks.append(img_doc)

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./db"
    )
    print(f"Done! Stored {len(chunks)} chunks including image descriptions.")
    return vectorstore

if __name__ == "__main__":
    ingest_pdf("data/sample.pdf")