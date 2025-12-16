# rag_batch_judge_fixed.py
# Batch-only: RAG (Gemini + Chroma) + LLM-as-a-Judge (Reference-based with Gemini)
# Robust JSON-only judge + parsing across response parts

import os, sys, json, re, time, base64
import pandas as pd

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

import google.generativeai as genai

# ========= Config =========
load_dotenv()  # חשוב: לפני os.getenv

# Data / Vector DB
DATA_DIR       = os.getenv("RAG_DATA_DIR", "data")           # תיקיית PDFים
PERSIST_DIR    = os.getenv("RAG_PERSIST_DIR", "chroma_db")   # איפה לשמור Chroma
COLLECTION     = os.getenv("RAG_COLLECTION", "enterprise_rag")
TOP_K          = int(os.getenv("RAG_TOPK", "6"))
REBUILD_INDEX  = os.getenv("RAG_REBUILD_INDEX", "0") == "1"  # הכרחה לבנות מחדש

# Models
EMB_MODEL      = os.getenv("RAG_EMB_MODEL", "models/text-embedding-004")
LLM_MODEL      = os.getenv("RAG_LLM_MODEL", "gemini-2.5-flash")
TEMPERATURE    = float(os.getenv("RAG_TEMP", "0.7"))

# Judge (Reference-based)
JUDGE_MODEL        = os.getenv("RAG_JUDGE_MODEL", "gemini-2.5-flash")
JUDGE_TEMPERATURE  = float(os.getenv("RAG_JUDGE_TEMP", "0.0"))
JUDGE_DEBUG        = bool(int(os.getenv("JUDGE_DEBUG", "0")))

# CSV I/O
CSV_IN   = os.getenv("JUDGE_IN_CSV", "judge/scoring.csv")
CSV_OUT  = os.getenv("JUDGE_OUT_CSV", "judge/scoring_scored.csv")

# API Key
API_KEY = os.getenv("GOOGLE_API_KEY", None)
if not API_KEY:
    print("ERROR: missing GOOGLE_API_KEY"); sys.exit(1)
genai.configure(api_key=API_KEY)

# ========= Judge Prompt (Reference-based) =========
REF_BASED_PROMPT = """You are an impartial evaluator for a RAG chatbot.
Evaluate the MODEL_ANSWER strictly against the given QUESTION and CONTEXT and the REFERENCE_ANSWER.

QUESTION:
{question}

CONTEXT (retrieved passages; you must not add external knowledge):
{context}

REFERENCE_ANSWER:
{reference_answer}

MODEL_ANSWER:
{model_answer}

Instructions:
1) Rate on a 1-5 scale (integers only):
   - accuracy: factual alignment with the reference.
   - faithfulness: grounded strictly in the provided context (no hallucinations).
   - relevance: directly answers the question.
   - completeness: covers the key points.
   - clarity: clarity and structure.
2) Determine if the model answer is equivalent to the reference answer (true/false).
3) Respond with ONLY a valid JSON object (no markdown, no backticks), using EXACTLY this schema:
{{
  "accuracy": <1-5>,
  "faithfulness": <1-5>,
  "relevance": <1-5>,
  "completeness": <1-5>,
  "clarity": <1-5>,
  "equivalent_to_reference": true/false,
  "final_score": <float between 1 and 5>,
  "reasoning": "<2-3 short sentences explaining the score>"
}}
Compute final_score as the arithmetic mean of the five criteria, rounded to one decimal place.
"""

# ========= Utils =========
def _coerce_scores(d: dict) -> dict:
    """Coerce types and compute final_score if missing."""
    def to_int(x):
        try: return int(x)
        except: return x
    def to_float(x):
        try: return float(x)
        except: return x

    keys_i = ["accuracy","faithfulness","relevance","completeness","clarity"]
    for k in keys_i:
        if k in d: d[k] = to_int(d[k])

    if "equivalent_to_reference" in d and isinstance(d["equivalent_to_reference"], str):
        d["equivalent_to_reference"] = d["equivalent_to_reference"].strip().lower() in ("true","1","yes")

    if "final_score" in d:
        d["final_score"] = to_float(d["final_score"])

    # compute final_score if missing
    if "final_score" not in d and all(k in d for k in keys_i):
        vals = [float(d[k]) for k in keys_i if isinstance(d[k], (int,float))]
        if vals:
            d["final_score"] = round(sum(vals)/len(vals), 1)
    return d

def _extract_json_from_resp(resp) -> dict:
    """
    Robustly extract JSON from Gemini response:
    - Prefer application/json parts
    - Else try text parts
    - Else fallback to resp.text
    - Handle malformed JSON with better error messages
    """
    # 1) candidates/parts
    try:
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content: continue
            parts = getattr(content, "parts", []) or []
            for p in parts:
                # inline JSON blob
                inline = getattr(p, "inline_data", None)
                if inline and getattr(inline, "mime_type", "") == "application/json":
                    data = getattr(inline, "data", b"")
                    if isinstance(data, bytes):
                        txt = data.decode("utf-8", "ignore")
                    else:
                        # some SDKs store base64 in string
                        try:
                            txt = base64.b64decode(data).decode("utf-8", "ignore")
                        except Exception:
                            txt = str(data)
                    return json.loads(txt)
                # text JSON
                t = getattr(p, "text", None)
                if t and "{" in t:
                    json_str = _pluck_json_block(t)
                    try:
                        return json.loads(json_str)
                    except json.JSONDecodeError as e:
                        # Log the problematic JSON for debugging
                        print(f"[debug] JSON decode error in part text: {e}")
                        print(f"[debug] Problematic JSON: {repr(json_str[:200])}")
                        raise
    except Exception as e:
        if "JSON decode error" not in str(e):
            print(f"[debug] Error processing parts: {e}")
    
    # 2) resp.text
    raw = (getattr(resp, "text", None) or "").strip()
    if raw:
        try:
            json_str = _pluck_json_block(raw)
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"[debug] JSON decode error in resp.text: {e}")
            print(f"[debug] Raw response: {repr(raw[:300])}")
            print(f"[debug] Extracted JSON: {repr(json_str[:200])}")
            raise
    
    raise ValueError("No JSON found in judge response")

def _pluck_json_block(s: str) -> str:
    """Return the innermost JSON object from a string (handles code fences)."""
    # Clean up the string first
    s = s.strip()
    
    # try fenced first
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.S) or re.search(r"```\s*(\{.*?\})\s*```", s, re.S)
    if m:
        return m.group(1).strip()
    
    # Handle malformed JSON that starts with newlines and quotes
    # Look for patterns like '\n "accuracy"' and try to fix them
    if s.startswith('\n') or s.startswith(' '):
        # Try to find the first complete JSON object
        brace_start = s.find('{')
        if brace_start != -1:
            s = s[brace_start:]
    
    # Find the outermost {...} but be more careful about matching
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(s):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                return s[start_idx:i+1]
    
    # Fallback: try regex
    m = re.search(r"\{.*\}", s, re.S)
    if m:
        return m.group(0).strip()
    
    # last resort: return as-is (will fail json.loads upstream)
    return s.strip()

def _detect_columns(df: pd.DataFrame):
    """Try to detect question/reference columns by common aliases."""
    cols_lower = [c.lower() for c in df.columns]
    def pick(candidates):
        for name in candidates:
            if name in cols_lower:
                return df.columns[cols_lower.index(name)]
        return None
    q_col  = pick(["question","prompt","query","שאלה"])
    ref_col= pick(["reference_answer","reference","expected","answer","expected_answer","תשובה"])
    if not q_col or not ref_col:
        raise ValueError(f"Could not detect question/reference columns. Found: {df.columns.tolist()}")
    return q_col, ref_col

def sources_as_str(docs) -> str:
    """Return unique sources as a compact string (source p.page | …)."""
    out, seen = [], set()
    for d in docs:
        src = d.metadata.get("source", "")
        page = d.metadata.get("page", "")
        s = f"{src}{' p.' + str(page) if page != '' else ''}"
        if src and s not in seen:
            seen.add(s)
            out.append(s)
    return " | ".join(out)

# ========= Build / Load Vector Store =========
def load_documents(data_dir: str):
    if not os.path.isdir(data_dir):
        return []
    try:
        return DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader).load()
    except Exception as e:
        print(f"[warn] failed loading PDFs from {data_dir}: {e}")
        return []

def chunk_documents(docs, chunk_size=800, overlap=120):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_documents(docs)

def build_or_load_chroma(persist_dir: str, collection: str, emb_model: str, rebuild: bool):
    emb = GoogleGenerativeAIEmbeddings(model=emb_model, google_api_key=API_KEY)

    if rebuild or (not os.path.exists(persist_dir) or not os.listdir(persist_dir)):
        # (Re)build
        docs = load_documents(DATA_DIR)
        chunks = chunk_documents(docs) if docs else []
        if not chunks:
            print("[warn] no documents found; index will be empty.")
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=emb,
            collection_name=collection,
            persist_directory=persist_dir,
        )
        return vectordb
    else:
        # Load existing
        return Chroma(
            collection_name=collection,
            persist_directory=persist_dir,
            embedding_function=emb
        )

# ========= Generation & Judging =========
def generate_answer(question: str, context: str) -> str:
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=TEMPERATURE, google_api_key=API_KEY)
    prompt = ChatPromptTemplate.from_template(
        "You are an enterprise assistant. Answer ONLY from the context. "
        "If not found, say you don't know. Add short citations like (source, page) when possible.\n\n"
        "Context:\n{context}\n\nQuestion:\n{question}"
    ).format(context=context, question=question)
    return llm.invoke(prompt).content.strip()

def judge_reference_based(question: str, context: str, model_answer: str, reference_answer: str,
                          model_name: str = None, temperature: float = 0.0, debug=False) -> dict:
    """
    Gemini judge that enforces JSON output and extracts it safely from response parts.
    Retries transient malformation up to 3 times.
    """
    model_name = model_name or os.getenv("RAG_JUDGE_MODEL", "gemini-2.5-flash")
    prompt = REF_BASED_PROMPT.format(
        question=question,
        context=context,
        model_answer=model_answer,
        reference_answer=reference_answer
    )
    if debug:
        print(f"[debug] Judge prompt:\n{prompt}")

    gen_model = genai.GenerativeModel(model_name)

    # Try with JSON mode first
    gen_cfg = {
        "temperature": temperature,
        "response_mime_type": "application/json"
    }

    last_err = None
    for attempt in range(3):
        try:
            resp = gen_model.generate_content(prompt, generation_config=gen_cfg)
            if debug:
                # Print first part briefly for debugging
                raw_dbg = getattr(resp, "text", None)
                if raw_dbg:
                    print(f"[debug] judge response: {repr(raw_dbg[:200])}")
            scores = _extract_json_from_resp(resp)
            # validate required keys
            required = ["accuracy","faithfulness","relevance","completeness","clarity",
                        "equivalent_to_reference","final_score","reasoning"]
            missing = [k for k in required if k not in scores]
            if missing:
                # Try soft-normalization of keys if model varied
                lowered = {str(k).strip().lower(): v for k,v in scores.items()}
                alias_map = {
                    "equivalent_to_reference": ["equivalent","match","matches_reference","equivalent_to_the_reference"]
                }
                for need, aliases in alias_map.items():
                    if need not in scores:
                        for a in aliases:
                            if a in lowered:
                                scores[need] = lowered[a]
                                break
                missing = [k for k in required if k not in scores]
                if missing:
                    raise ValueError(f"missing keys: {missing} | got keys: {list(scores.keys())}")

            scores = _coerce_scores(scores)
            return scores
        except Exception as e:
            last_err = e
            if debug:
                print(f"[debug] judge attempt {attempt+1} failed: {e}")
            
            # If JSON mode fails, try without it as fallback
            if attempt == 2:  # Last attempt
                try:
                    if debug:
                        print("[debug] Trying fallback without JSON mode...")
                    gen_cfg_fallback = {"temperature": temperature}
                    resp = gen_model.generate_content(prompt, generation_config=gen_cfg_fallback)
                    if debug:
                        raw_dbg = getattr(resp, "text", None)
                        if raw_dbg:
                            print("[debug] fallback text head:", raw_dbg[:200])
                    scores = _extract_json_from_resp(resp)
                    # Same validation logic
                    required = ["accuracy","faithfulness","relevance","completeness","clarity",
                                "equivalent_to_reference","final_score","reasoning"]
                    missing = [k for k in required if k not in scores]
                    if missing:
                        lowered = {str(k).strip().lower(): v for k,v in scores.items()}
                        alias_map = {
                            "equivalent_to_reference": ["equivalent","match","matches_reference","equivalent_to_the_reference"]
                        }
                        for need, aliases in alias_map.items():
                            if need not in scores:
                                for a in aliases:
                                    if a in lowered:
                                        scores[need] = lowered[a]
                                        break
                        missing = [k for k in required if k not in scores]
                        if missing:
                            raise ValueError(f"missing keys: {missing} | got keys: {list(scores.keys())}")
                    scores = _coerce_scores(scores)
                    return scores
                except Exception as fallback_err:
                    if debug:
                        print(f"[debug] fallback also failed: {fallback_err}")
                    last_err = fallback_err
            
            time.sleep(0.6)
            continue
    raise last_err

# ========= Main (Batch only) =========
def main():
    # Load/Build vector store
    vectordb = build_or_load_chroma(PERSIST_DIR, COLLECTION, EMB_MODEL, REBUILD_INDEX)

    # Read CSV
    if not os.path.exists(CSV_IN):
        print(f"ERROR: CSV file not found: {CSV_IN}")
        sys.exit(2)
    df = pd.read_csv(CSV_IN, encoding="utf-8-sig")
    q_col, ref_col = _detect_columns(df)

    results = []
    for i, row in df.iterrows():
        question = str(row[q_col]).strip()
        reference_answer = str(row[ref_col]).strip()

        # Retrieve
        relevant = vectordb.similarity_search(question, k=TOP_K)
        context = "\n\n".join(d.page_content for d in relevant)

        # Generate
        try:
            model_answer = generate_answer(question, context)
        except Exception as e:
            model_answer = ""
            judge_scores = {"error": f"generation_failed: {e}"}
            results.append({
                "question": question,
                "reference_answer": reference_answer,
                "model_answer": model_answer,
                "sources": sources_as_str(relevant),
                **judge_scores
            })
            continue

        # Judge (reference-based)
        try:
            judge_scores = judge_reference_based(
                question, context, model_answer, reference_answer,
                model_name=JUDGE_MODEL,
                temperature=JUDGE_TEMPERATURE,
                debug=JUDGE_DEBUG
            )
        except Exception as e:
            judge_scores = {
                "error": f"judge_failed: {e}",
                "accuracy": 0, "faithfulness": 0, "relevance": 0, "completeness": 0, "clarity": 0,
                "equivalent_to_reference": False, "final_score": 0, "reasoning": ""
            }
            print(f"[warn] judging failed for row {i}: {e}")

        results.append({
            "question": question,
            "reference_answer": reference_answer,
            "model_answer": model_answer,
            "sources": sources_as_str(relevant),
            **judge_scores
        })

    out_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True) if os.path.dirname(CSV_OUT) else None
    out_df.to_csv(CSV_OUT, index=False, encoding="utf-8-sig")
    print(f"[ok] wrote: {CSV_OUT}")
    # Summary
    if "final_score" in out_df.columns:
        try:
            print(out_df["final_score"].astype(float).describe().to_string())
        except Exception:
            pass

if __name__ == "__main__":
    main()
