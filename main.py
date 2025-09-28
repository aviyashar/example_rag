# RAG with Gemini + Chroma (no Vertex AI) — single-file script
import os, sys
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import google.generativeai as genai

# ===== Config =====
DATA_DIR = os.getenv("RAG_DATA_DIR", "data")               # folder with PDFs
PERSIST_DIR = os.getenv("RAG_PERSIST_DIR", "chroma_db")    # Chroma persistence
COLLECTION = os.getenv("RAG_COLLECTION", "enterprise_rag")
QUESTION = os.getenv("RAG_QUESTION", "תסכם לי את המסמך")
TOP_K = int(os.getenv("RAG_TOPK", "4"))
EMB_MODEL = os.getenv("RAG_EMB_MODEL", "models/text-embedding-004")
LLM_MODEL = os.getenv("RAG_LLM_MODEL", "gemini-2.5-flash")  # or gemini-1.5-pro
TEMPERATURE = float(os.getenv("RAG_TEMP", "0.2"))

# ===== API Key =====
api_key = "AviYashar"
if not api_key:
    print("ERROR: GOOGLE_API_KEY env var is missing."); sys.exit(1)
genai.configure(api_key=api_key)

# ===== Load Documents =====
docs = []
if os.path.isdir(DATA_DIR):
    try:
        docs = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    except Exception as e:
        print(f"[warn] failed loading PDFs from {DATA_DIR}: {e}")
if not docs:  # fallback so script runs even without files
    sample = ("Company Vacation Policy:\n"
              "- Full-time employees: 18 days/year.\n"
              "- Contractors: 10 business days/year after 3 months.\n"
              "- Carryover up to 5 days with manager approval.")
    docs = [Document(page_content=sample, metadata={"source": "sample_policy.txt", "page": 1})]

# ===== Chunking =====
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
chunks = splitter.split_documents(docs)

# ===== Embeddings + Chroma Vector Store =====
emb = GoogleGenerativeAIEmbeddings(model=EMB_MODEL, google_api_key=api_key)

# Build (or rebuild) the index each run; persisted to disk
vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=emb,
    collection_name=COLLECTION,
    persist_directory=PERSIST_DIR,
)
vectordb.persist()

# ===== Query + simple guardrail =====
user_q = QUESTION.strip()
print(f"\n[query] {user_q}\n")
blocked_terms = {"private key", "customer ssn", "password dump", "national id"}
if any(t in user_q.lower() for t in blocked_terms):
    print({"answer": "Sorry, I can’t share that information.", "sources": []}); sys.exit(0)

# ===== Retrieve =====
relevant = vectordb.similarity_search(user_q, k=TOP_K)
# Example with metadata filter:
# relevant = vectordb.similarity_search(user_q, k=TOP_K, filter={"dept": "HR"})

# ===== Augment (prompt with context) =====
context = "\n\n".join(d.page_content for d in relevant)
prompt = ChatPromptTemplate.from_template(
    "You are an enterprise assistant. Answer ONLY from the context. "
    "If not found, say you don't know. Add short citations like (source, page) when possible.\n\n"
    "Context:\n{context}\n\nQuestion:\n{question}"
).format(context=context, question=user_q)

# ===== Generate (Gemini) =====
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE, google_api_key=api_key)
answer = llm.invoke(prompt).content.strip()

# ===== Collect Sources =====
sources = []
for d in relevant:
    src, page = d.metadata.get("source",""), d.metadata.get("page","")
    if src: sources.append(f"{src}{' p.'+str(page) if page else ''}")
# de-dup while preserving order
seen, dedup = set(), []
for s in sources:
    if s and s not in seen: seen.add(s); dedup.append(s)

# ===== Output =====
print("\n[answer]\n" + answer + "\n")
print("[sources]"); [print("- " + s) for s in dedup]
print("\n[payload]\n", {"answer": answer, "sources": dedup})
