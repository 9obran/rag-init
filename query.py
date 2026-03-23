"""
RAG Query Pipeline
Retrieves relevant context from ChromaDB and generates answers using Groq's free LLM API.
"""

import os
import sys
from pathlib import Path

# Load environment variables securely
from dotenv import load_dotenv
load_dotenv()

# Configuration
CHROMA_PERSIST_DIR = Path("chroma_db")

# Get API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Imports for RAG pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def check_setup():
    """Verify that the environment is properly configured."""
    print("Checking setup...")
    
    # Check API key
    if not GROQ_API_KEY:
        print("\n" + "!" * 60)
        print("ERROR: GROQ_API_KEY not found in environment!")
        print("!" * 60)
        print("\nTo fix this:")
        print("1. Copy .env.example to .env:")
        print("   cp .env.example .env")
        print("2. Get a FREE API key from https://console.groq.com")
        print("3. Add your key to the .env file")
        print("\nNote: .env is gitignored for security - it won't be committed!")
        print("!" * 60)
        return False
    
    # Check if ChromaDB exists
    if not CHROMA_PERSIST_DIR.exists():
        print(f"\nERROR: ChromaDB not found at {CHROMA_PERSIST_DIR}")
        print("Please run ingest.py first to create the vector database:")
        print("   python ingest.py")
        return False
    
    print("Setup verified: API key and vector database found.")
    return True


def load_vectorstore():
    """
    Load the ChromaDB vector store with HuggingFace embeddings.
    
    Returns:
        Chroma vector store instance
    """
    print(f"Loading vector database from {CHROMA_PERSIST_DIR}...")
    
    # Use the same free, local embedding model as during ingestion
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    vectorstore = Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=embeddings
    )
    
    print(f"Vector database loaded: {vectorstore._collection.count()} documents")
    return vectorstore


def setup_llm():
    """
    Initialize the Groq LLM (free tier available).
    
    Groq provides free access to open-source models like Llama 3.1, Mixtral.
    This is a cost-effective alternative to OpenAI for portfolio projects.
    
    Returns:
        Groq LLM instance
    """
    print("Initializing Groq LLM (Llama 3.1 8B - free tier)...")
    
    try:
        from langchain_groq import ChatGroq
        
        llm = ChatGroq(
            model_name="llama-3.1-8b-instant",  # Fast, capable, free tier available
            temperature=0.2,  # Lower temperature for factual answers
            max_tokens=1024,
            groq_api_key=GROQ_API_KEY
        )
        
        print("Groq LLM initialized successfully")
        return llm
        
    except ImportError:
        print("\nERROR: langchain-groq not installed.")
        print("Install with: pip install langchain-groq")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR initializing Groq: {e}")
        print("Please verify your GROQ_API_KEY in .env")
        sys.exit(1)


def create_rag_chain(vectorstore, llm):
    """
    Create the RAG chain combining retrieval and generation.
    Uses a simple custom implementation compatible with installed LangChain version.
    
    Args:
        vectorstore: Chroma vector store
        llm: Language model instance
        
    Returns:
        SimpleRAGChain class instance
    """
    from langchain_core.prompts import ChatPromptTemplate
    
    # Create a simple RAG class
    class SimpleRAGChain:
        def __init__(self, retriever, llm):
            self.retriever = retriever
            self.llm = llm
            
        def invoke(self, inputs):
            question = inputs.get("input", inputs.get("query", ""))
            
            # Retrieve relevant documents
            docs = self.retriever.invoke(question)
            
            # Format context from retrieved docs
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Create prompt with context and question
            prompt = f"""You are a helpful assistant answering questions based on the Wintershall internship report.
Use ONLY the following context to answer the question. If you don't know the answer based on the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {question}

Answer:"""
            
            # Call LLM
            response = self.llm.invoke(prompt)
            
            return {
                "answer": response.content if hasattr(response, 'content') else str(response),
                "context": docs
            }
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )
    
    qa_chain = SimpleRAGChain(retriever, llm)
    
    return qa_chain


def ask_question(qa_chain, question: str):
    """
    Execute a RAG query and return the answer.
    
    Args:
        qa_chain: Configured RAG chain
        question: User question string
        
    Returns:
        Dictionary containing answer and source documents
    """
    print(f"\nQuestion: {question}")
    print("-" * 60)
    
    result = qa_chain.invoke({"input": question})
    
    answer = result["answer"]
    sources = result.get("context", [])
    
    print(f"Answer: {answer}")
    print()
    
    # Show source chunks (for transparency)
    if sources:
        print("Sources (top relevant chunks):")
        for i, doc in enumerate(sources[:2], 1):
            preview = doc.page_content[:150].replace('\n', ' ')
            print(f"  [{i}] ...{preview}...")
    
    return {"answer": answer, "sources": sources}


def interactive_mode(qa_chain):
    """Run interactive Q&A session."""
    print("\n" + "=" * 60)
    print("INTERACTIVE RAG MODE")
    print("=" * 60)
    print("Type your questions about the Wintershall internship report.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 60)
    
    while True:
        print()
        question = input("Question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not question:
            continue
        
        try:
            ask_question(qa_chain, question)
        except Exception as e:
            print(f"Error: {e}")


def run_test_questions(qa_chain):
    """Run predefined test questions for portfolio documentation."""
    test_questions = [
        "What company did the intern work for and what do they do?",
        "What was the main focus of the internship?",
        "What specific technical work was completed during the internship?",
        "What software and tools were used during the internship?",
        "What are the key learnings from this internship?"
    ]
    
    print("\n" + "=" * 60)
    print("RUNNING TEST QUESTIONS FOR PORTFOLIO")
    print("=" * 60)
    
    results = []
    for question in test_questions:
        result = ask_question(qa_chain, question)
        results.append({
            "question": question,
            "answer": result["answer"]
        })
        print("\n" + "-" * 60)
    
    return results


def main():
    """Main query pipeline execution."""
    print("=" * 60)
    print("RAG QUERY PIPELINE")
    print("=" * 60)
    print()
    
    # Verify setup
    if not check_setup():
        sys.exit(1)
    
    # Load components
    vectorstore = load_vectorstore()
    llm = setup_llm()
    
    # Create RAG chain
    print("Creating RAG chain...")
    qa_chain = create_rag_chain(vectorstore, llm)
    print("RAG pipeline ready!")
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Run test mode for portfolio documentation
        results = run_test_questions(qa_chain)
        return results
    else:
        # Interactive mode
        interactive_mode(qa_chain)


if __name__ == "__main__":
    main()
