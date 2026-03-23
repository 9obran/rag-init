"""
RAG Data Ingestion Pipeline
Loads documents, splits into chunks, and stores in ChromaDB vector database.
"""

import os
import sys
from pathlib import Path

# Load environment variables securely
from dotenv import load_dotenv
load_dotenv()

# NOTE: No API key required for ingestion - we use local HuggingFace embeddings
# The GROQ_API_KEY is only needed in query.py for the LLM generation step

# LangChain imports for document processing
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Configuration
DATA_DIR = Path("data")
CHROMA_PERSIST_DIR = Path("chroma_db")
CHUNK_SIZE = 500  # Characters per chunk
CHUNK_OVERLAP = 50  # Overlap between chunks for context continuity


def load_documents(data_dir: Path):
    """
    Load all text documents from the data directory.
    
    Args:
        data_dir: Path to directory containing source documents
        
    Returns:
        List of Document objects
    """
    print(f"Loading documents from {data_dir}...")
    
    # Use DirectoryLoader for multiple files, or TextLoader for single file
    if data_dir.exists():
        loader = DirectoryLoader(
            str(data_dir),
            glob="*.txt",
            loader_cls=TextLoader
        )
        documents = loader.load()
        print(f"Loaded {len(documents)} document(s)")
        return documents
    else:
        print(f"Error: Data directory {data_dir} not found!")
        sys.exit(1)


def split_documents(documents):
    """
    Split documents into overlapping chunks for better retrieval.
    
    RecursiveCharacterTextSplitter tries to split on natural boundaries
    (paragraphs, sentences) while respecting chunk size.
    
    Args:
        documents: List of Document objects
        
    Returns:
        List of chunked Document objects
    """
    print(f"Splitting documents into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]  # Priority order for splitting
    )
    
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Show sample chunk sizes
    if chunks:
        sizes = [len(chunk.page_content) for chunk in chunks[:5]]
        print(f"Sample chunk sizes: {sizes}")
    
    return chunks


def create_embeddings():
    """
    Initialize the embedding model.
    
    Using HuggingFace's free, local embeddings (all-MiniLM-L6-v2) 
    instead of OpenAI to keep costs at zero and data local.
    
    Returns:
        HuggingFaceEmbeddings instance
    """
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    print("This is a FREE, local model - no API calls required for embeddings.")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},  # Use CPU for portability
        encode_kwargs={'normalize_embeddings': True}
    )
    
    return embeddings


def store_in_chroma(chunks, embeddings, persist_dir: Path):
    """
    Store document chunks in ChromaDB vector database.
    
    Args:
        chunks: List of chunked Document objects
        embeddings: Embedding model instance
        persist_dir: Directory to persist ChromaDB data
    """
    print(f"Storing {len(chunks)} chunks in ChromaDB at {persist_dir}...")
    
    # Remove existing database if present (fresh start)
    if persist_dir.exists():
        import shutil
        shutil.rmtree(persist_dir)
        print("Removed existing ChromaDB for fresh start")
    
    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_dir)
    )
    
    # Persist to disk
    vectorstore.persist()
    
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
    print(f"Database location: {persist_dir.absolute()}")
    
    return vectorstore


def main():
    """Main ingestion pipeline execution."""
    print("=" * 60)
    print("RAG DATA INGESTION PIPELINE")
    print("=" * 60)
    print()
    
    # Step 1: Load documents
    documents = load_documents(DATA_DIR)
    
    # Step 2: Split into chunks
    chunks = split_documents(documents)
    
    # Step 3: Create embeddings (free, local model)
    embeddings = create_embeddings()
    
    # Step 4: Store in ChromaDB
    vectorstore = store_in_chroma(chunks, embeddings, CHROMA_PERSIST_DIR)
    
    print()
    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total chunks indexed: {len(chunks)}")
    print(f"Vector store location: {CHROMA_PERSIST_DIR.absolute()}")
    print()
    print("You can now run query.py to ask questions!")
    
    return vectorstore


if __name__ == "__main__":
    main()
