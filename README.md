# RAG System with Google Gemini + ChromaDB

A collection of Retrieval-Augmented Generation (RAG) implementations using Google's Gemini AI models and ChromaDB vector database. This project demonstrates different RAG patterns from basic single-question systems to advanced LLM-as-a-Judge evaluation frameworks.

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Scripts Overview](#scripts-overview)
- [Usage](#usage)
- [Project Structure](#project-structure)

## Features

- PDF document ingestion and processing
- Vector embeddings with Google's text-embedding-004
- Multiple RAG implementations (single-question, multi-question, batch processing)
- LLM-as-a-Judge evaluation (reference-free and reference-based)
- Configurable via environment variables
- ChromaDB for efficient vector storage and retrieval

## Prerequisites

- Python 3.8 or higher
- Google AI API Key (for Gemini models)
- PDF documents for knowledge base

## Installation

1. Clone or download this repository

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

The `requirements.txt` includes:
- LangChain ecosystem (langchain, langchain-core, langchain-community)
- Google Generative AI integrations
- ChromaDB for vector storage
- PDF processing utilities (pypdf)
- Data processing (pandas)
- Environment management (python-dotenv)

## Configuration

### 1. Create a `.env` file

Create a `.env` file in the root directory with your Google API key:

```bash
# Required: Google AI API Key
GOOGLE_API_KEY=your_google_api_key_here

# Optional: Override default configurations
RAG_DATA_DIR=data
RAG_PERSIST_DIR=chroma_db
RAG_COLLECTION=enterprise_rag
RAG_TOPK=6
RAG_EMB_MODEL=models/text-embedding-004
RAG_LLM_MODEL=gemini-2.5-flash
RAG_TEMP=0.8
RAG_QUESTION="What is big data?"

# Optional: Judge model configuration (for rag_3 and rag_4)
RAG_JUDGE_MODEL=gemini-2.5-flash
RAG_JUDGE_TEMP=0.0
JUDGE_DEBUG=0

# Optional: CSV paths for batch processing (rag_3)
JUDGE_IN_CSV=judge/scoring.csv
JUDGE_OUT_CSV=judge/scoring_scored.csv
RAG_REBUILD_INDEX=0
```

### 2. Get Google AI API Key

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### 3. Add PDF Documents

Place your PDF documents in the `data/` directory. The system will automatically:
- Load all PDFs from the directory
- Split them into chunks
- Create embeddings
- Store them in ChromaDB

## Scripts Overview

### `rag_1.py` - Basic Single-Question RAG

**Purpose**: Simple RAG implementation that answers a single predefined question.

**What it does**:
- Loads PDFs from the `data/` directory
- Creates vector embeddings using Google's text-embedding-004
- Builds a ChromaDB vector store
- Answers a single question using context from the documents
- Outputs the answer with source citations

**Use case**: Quick testing, demos, or single-query scenarios

---

### `rag_2_multi_question.py` - Interactive Multi-Question RAG

**Purpose**: Interactive RAG system that allows multiple questions in a loop.

**What it does**:
- Same setup as rag_1.py (loads PDFs, creates embeddings)
- Enters an interactive loop where you can ask unlimited questions
- Each question retrieves relevant context and generates an answer
- Includes basic guardrails to block sensitive queries
- Type 'exit' to quit

**Use case**: Interactive exploration of documents, testing different queries

---

### `rag_3_llm-jadge-wr-based.py` - Batch Processing with Reference-Based Judge

**Purpose**: Batch evaluation system with LLM-as-a-Judge using reference answers.

**What it does**:
- Reads questions and reference answers from a CSV file (`judge/scoring.csv`)
- Processes each question through the RAG system
- Generates answers based on document context
- Uses Gemini as a judge to evaluate answers against reference answers
- Scores on 5 criteria:
  - **Accuracy**: Factual alignment with reference
  - **Faithfulness**: Grounded in context (no hallucinations)
  - **Relevance**: Directly answers the question
  - **Completeness**: Covers key points
  - **Clarity**: Clear and well-structured
- Outputs results to `judge/scoring_scored.csv` with scores and reasoning

**CSV Format** (judge/scoring.csv):
```csv
question,reference_answer
"What is big data?","Big data refers to data sets that are too large or complex for traditional software."
"What are the 3 Vs of big data?","Volume, Variety, and Velocity"
```

**Use case**: Systematic evaluation of RAG performance, quality assurance, benchmark testing

---

### `rag_4_llm-jadge-wor.py` - Interactive RAG with Reference-Free Judge

**Purpose**: Interactive RAG with automatic answer quality evaluation (no reference needed).

**What it does**:
- Interactive multi-question system like rag_2.py
- After each answer, automatically evaluates it using LLM-as-a-Judge
- Scores on 4 criteria (reference-free):
  - **Faithfulness**: Grounded strictly in context
  - **Relevance**: Addresses the question
  - **Completeness**: Covers key points
  - **Clarity**: Clear and well-structured
- Displays scores and reasoning after each answer
- No reference answers required

**Use case**: Real-time answer quality monitoring, testing with immediate feedback

## Usage

### Running rag_1.py (Single Question)

```bash
python3 rag_1.py
```

To change the question, edit line 20 in the file or set environment variable:
```bash
RAG_QUESTION="Your question here" python3 rag_1.py
```

**Output**:
```
[query] What is big data?

[answer]
Big data primarily refers to data sets that are too large or complex...

[sources]
- data/Big data - Wikipedia.pdf

[payload]
{'answer': '...', 'sources': ['...']}
```

---

### Running rag_2_multi_question.py (Interactive)

```bash
python3 rag_2_multi_question.py
```

**Interactive session**:
```
Enter your questions (type 'exit' to quit):
Your question: What is big data?

[answer]
Big data primarily refers to...

[sources]
- data/Big data - Wikipedia.pdf

Your question: What are the challenges?

[answer]
...

Your question: exit
Exiting...
```

---

### Running rag_3_llm-jadge-wr-based.py (Batch with Judge)

1. Create your input CSV file at `judge/scoring.csv`:
```csv
question,reference_answer
"What is big data?","Big data refers to extremely large datasets"
"What is machine learning?","Machine learning is a subset of AI"
```

2. Run the script:
```bash
python3 rag_3_llm-jadge-wr-based.py
```

**Output**:
```
[ok] wrote: judge/scoring_scored.csv
count    2.00
mean     4.85
std      0.15
min      4.70
25%      4.78
50%      4.85
75%      4.93
max      5.00
```

3. Check results in `judge/scoring_scored.csv`:
```csv
question,reference_answer,model_answer,sources,accuracy,faithfulness,relevance,completeness,clarity,equivalent_to_reference,final_score,reasoning
"What is big data?","...","...","data/file.pdf",5,5,5,4,5,true,4.8,"The answer accurately..."
```

---

### Running rag_4_llm-jadge-wor.py (Interactive with Judge)

```bash
python3 rag_4_llm-jadge-wor.py
```

**Interactive session with evaluation**:
```
Enter your questions (type 'exit' to quit):
Your question: What is big data?

[answer]
Big data primarily refers to...

[sources]
- data/Big data - Wikipedia.pdf

[judge]
{'faithfulness': 5, 'relevance': 5, 'completeness': 4, 'clarity': 5,
 'final_score': 4.8, 'reasoning': 'The answer is faithful to the context...'}

Your question: exit
Exiting...
```

## Project Structure

```
example_rag/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables (create this)
├── .gitignore                        # Git ignore file
│
├── rag_1.py                          # Basic single-question RAG
├── rag_2_multi_question.py           # Interactive multi-question RAG
├── rag_3_llm-jadge-wr-based.py       # Batch processing with reference-based judge
├── rag_4_llm-jadge-wor.py            # Interactive with reference-free judge
│
├── data/                             # Your PDF documents go here
│   └── Big data - Wikipedia.pdf
│
├── chroma_db/                        # ChromaDB vector database (auto-generated)
│   └── [vector store files]
│
└── judge/                            # Judge evaluation files (for rag_3)
    ├── scoring.csv                   # Input: questions & reference answers
    └── scoring_scored.csv            # Output: results with scores
```

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `GOOGLE_API_KEY` | **Required** | Your Google AI API key |
| `RAG_DATA_DIR` | `data` | Directory containing PDF files |
| `RAG_PERSIST_DIR` | `chroma_db` | ChromaDB storage directory |
| `RAG_COLLECTION` | `enterprise_rag` | ChromaDB collection name |
| `RAG_TOPK` | `6` | Number of relevant documents to retrieve |
| `RAG_EMB_MODEL` | `models/text-embedding-004` | Embedding model |
| `RAG_LLM_MODEL` | `gemini-2.5-flash` | LLM model for generation |
| `RAG_TEMP` | `0.8` | Temperature for generation (0.0-1.0) |
| `RAG_QUESTION` | varies | Default question for rag_1.py |
| `RAG_JUDGE_MODEL` | `gemini-2.5-flash` | Judge model (rag_3, rag_4) |
| `RAG_JUDGE_TEMP` | `0.0` | Judge temperature |
| `JUDGE_DEBUG` | `0` | Enable judge debug output (1=on) |
| `JUDGE_IN_CSV` | `judge/scoring.csv` | Input CSV for rag_3 |
| `JUDGE_OUT_CSV` | `judge/scoring_scored.csv` | Output CSV for rag_3 |
| `RAG_REBUILD_INDEX` | `0` | Force rebuild vector DB (1=on) |

## Troubleshooting

### "GOOGLE_API_KEY env var is missing"
- Make sure you created a `.env` file with your API key
- Verify the key is correct (no extra spaces or quotes)

### "No documents found"
- Check that PDF files are in the `data/` directory
- Ensure PDFs are readable and not corrupted

### Import errors
- Run `pip install -r requirements.txt` to install dependencies
- Make sure you're using Python 3.8+

### "I don't know" answers
- The system only answers from document context
- Add relevant PDFs to your `data/` directory
- Try questions that match your document content

## Advanced Tips

1. **Custom Questions**: Edit the `QUESTION` variable in rag_1.py or use environment variables

2. **Model Selection**: Use `gemini-2.5-pro` for better quality (slower, more expensive)

3. **Chunk Size**: Modify `chunk_size` and `chunk_overlap` in the scripts for different document types

4. **Retrieval Tuning**: Adjust `TOP_K` to retrieve more/fewer relevant chunks

5. **Temperature**: Lower temperature (0.0-0.3) for factual answers, higher (0.7-1.0) for creative responses

## License

This project is provided as-is for educational and demonstration purposes.
