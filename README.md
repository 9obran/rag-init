#  RAG init Project

A production ready Retrieval Augmented Generation (RAG) application built with Python, LangChain, and ChromaDB. This project demonstrates practical AI engineering skills using completely free LLM APIs.


## Overview

This RAG application allows you to:
- **Ingest** documents into a local vector database
- **Retrieve** relevant context using semantic search
- **Generate** answers using free LLM APIs (Groq)


## Tech Stack

| Component | Technology | Why I Chose It |
|-----------|------------|----------------|
| **Framework** | LangChain | Standard for RAG, extensive community support |
| **Vector DB** | ChromaDB | Free, local first, no external dependencies |
| **LLM** | Groq (Llama 3.1) | Free tier, fast inference, open source models |
| **Embeddings** | HuggingFace (all-MiniLM-L6-v2) | Free, local, no API calls for embeddings |

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Source Docs    │────▶│  Text Splitter  │────▶│  HuggingFace    │
│  (data/*.txt)   │     │  (500 char      │     │  Embeddings     │
│                 │     │   chunks)       │     │  (local)        │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Generated      │◀────│  Groq LLM       │◀────│  ChromaDB       │
│  Answer         │     │  (Llama 3.1)    │     │  (local         │
│                 │     │                 │     │   vector store) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
       ▲                                               ▲
       │                                               │
       └───────────────────────────────────────────────┘
                    User Question
```

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd rag-portfolio
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
cp .env.example .env
# Edit .env and add your free Groq API key
# Get one at: https://console.groq.com (free tier available)
```

### 3. Ingest Documents

```bash
python ingest.py
```

### 4. Run Queries

```bash
# Interactive mode
python query.py

# Test mode (for portfolio verification)
python query.py --test
```


## Sample Test Results

```
Question: What company did the intern work for?
Answer: The intern worked at Wintershall Dea GmbH, one of Europe's 
leading independent natural gas and crude oil producers.

Question: What was the main focus of the internship?
Answer: The internship focused on geological data analysis, reservoir 
characterization, and operational support for North Sea gas production 
facilities over a 12-week period.
```


