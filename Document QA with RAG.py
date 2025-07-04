import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

# === Load .env credentials ===
load_dotenv()

# === Load PDF ===
pdf_path = "C:\\Users\\patet\\Downloads\\DSP\\DSP\\unsupervised.pdf"
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f'The file {pdf_path} does not exist.')

loader = PyPDFLoader(pdf_path)
documents = loader.load()
print(f"✅ Loaded {len(documents)} documents from {pdf_path}")

# === Split into Chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)
print(f"✅ Split into {len(texts)} chunks")

# === Embeddings ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Vector Store ===
vector_store = FAISS.from_documents(texts, embeddings)
print(f"✅ Created vector store with {len(texts)} chunks")

# === Load OpenRouter-compatible LLM ===
llm = ChatOpenAI(
    model_name="mistralai/mistral-7b-instruct",  # or "gpt-3.5-turbo"
    temperature=0.7,
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# === Create Retrieval Chain ===
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# === Run Query 1 ===
question1 = "What is the main topic of the document?"
result1 = qa_chain({"question": question1})
print("\n🔎 Question 1:", question1)
print("🧠 Answer:", result1['result'])

print("\n📄 Source Documents:")
for doc in result1['source_documents']:
    print(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")

# === Save Vector Store ===
vector_store.save_local("vector_store")

# === Load Vector Store ===
loaded_vector_store = FAISS.load_local("vector_store", embeddings)
print(f"\n✅ Reloaded vector store with {len(loaded_vector_store)} documents")

# === Run Query 2 ===
question2 = "What are the key concepts discussed in the document?"
result2 = qa_chain({"question": question2})
print("\n🔎 Question 2:", question2)
print("🧠 Answer:", result2['result'])

print("\n📄 Source Documents:")
for doc in result2['source_documents']:
    print(f"- {doc.metadata['source']}: {doc.page_content[:200]}...")
