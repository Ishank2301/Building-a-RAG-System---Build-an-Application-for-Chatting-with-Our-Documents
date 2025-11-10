import os
from dotenv import load_dotenv
import chromadb
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# Load environment variables
load_dotenv()

# 1️⃣ Embeddings using Ollama (local model)
ollama_ef = OllamaEmbeddings(model="nomic-embed-text")

# 2️⃣ Initialize Chroma client with persistence
chroma_path = os.getenv("CHROMA_PATH", "chroma_persistent_storage")
chroma_client = chromadb.PersistentClient(path=chroma_path)
collection_name = "ollama_qa_collection"
collection = chroma_client.get_or_create_collection(name=collection_name)

# 3️⃣ Local Ollama model for answering
llm = OllamaLLM(model="llama3")

# 4️⃣ Load documents
def load_documents_from_directory(directory_path):
    print("==== Loading documents from directory ====")
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".txt"):
            with open(os.path.join(directory_path, filename), "r", encoding="utf-8") as file:
                documents.append({"id": filename, "text": file.read()})
    return documents

# 5️⃣ Split documents into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

# 6️⃣ Load and chunk
directory_path = "./news_articles"
if not os.path.exists(directory_path):
    print("⚠️ Folder './news_articles' not found — create it and add .txt files.")
    exit()

documents = load_documents_from_directory(directory_path)
print(f"Loaded {len(documents)} documents")

chunked_documents = []
for doc in documents:
    chunks = split_text(doc["text"])
    print(f"==== Splitting '{doc['id']}' into {len(chunks)} chunks ====")
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id": f"{doc['id']}_chunk{i+1}", "text": chunk})

# 7️⃣ Generate embeddings
def get_ollama_embedding(text):
    print("==== Generating embeddings... ====")
    return ollama_ef.embed_query(text)

# 8️⃣ Upsert embeddings into Chroma
for i, doc in enumerate(chunked_documents, 1):
    print(f"=== [{i}/{len(chunked_documents)}] Generating embeddings... ===")
    embedding = get_ollama_embedding(doc["text"])
    collection.upsert(
        ids=[doc["id"]],
        documents=[doc["text"]],
        embeddings=[embedding],
    )

# 9️⃣ Query documents
def query_documents(question, n_results=3):
    query_embedding = get_ollama_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("=== Returning relevant chunks ===")
    return relevant_chunks

# 🔟 Generate response using Ollama
def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{question}"
    )

    print("=== Generating answer using Ollama ===")
    answer = llm.invoke(prompt)
    return answer

# 🧩 Run example query
question = "Tell me about Databricks"
relevant_chunks = query_documents(question)
answer = generate_response(question, relevant_chunks)
print("\n✅ Final Answer:\n", answer)
