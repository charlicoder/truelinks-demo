"""
Knowledge Base Module
Handles PDF extraction and vector store for QCS2024 documents.
Supports persistence to avoid rebuilding on every restart.
"""

import os
import json
import pickle
import fitz  # PyMuPDF
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
import faiss


class KnowledgeBase:
    """Manages QCS2024 document ingestion and semantic search with persistence."""
    
    # Default paths for persistence
    DEFAULT_DB_DIR = "kb_database"
    INDEX_FILE = "faiss_index.bin"
    CHUNKS_FILE = "chunks.json"
    META_FILE = "metadata.json"
    
    def __init__(self, docs_path: str = None, db_path: str = None):
        self.docs_path = docs_path or os.path.join(
            os.path.dirname(__file__), "QCS2024", "QCS 2024"
        )
        self.db_path = db_path or os.path.join(
            os.path.dirname(__file__), self.DEFAULT_DB_DIR
        )
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunks: List[Dict] = []
        self.index = None
        self.is_initialized = False
        
    def _get_index_path(self) -> str:
        return os.path.join(self.db_path, self.INDEX_FILE)
    
    def _get_chunks_path(self) -> str:
        return os.path.join(self.db_path, self.CHUNKS_FILE)
    
    def _get_meta_path(self) -> str:
        return os.path.join(self.db_path, self.META_FILE)
    
    def _db_exists(self) -> bool:
        """Check if the database files exist."""
        return (
            os.path.exists(self._get_index_path()) and 
            os.path.exists(self._get_chunks_path())
        )
    
    def save_to_disk(self) -> None:
        """Save the FAISS index and chunks to disk."""
        if not self.is_initialized:
            print("Warning: Cannot save uninitialized knowledge base")
            return
            
        # Create database directory
        os.makedirs(self.db_path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, self._get_index_path())
        print(f"Saved FAISS index to {self._get_index_path()}")
        
        # Save chunks as JSON
        with open(self._get_chunks_path(), 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(self.chunks)} chunks to {self._get_chunks_path()}")
        
        # Save metadata
        metadata = {
            "chunks_count": len(self.chunks),
            "docs_path": self.docs_path
        }
        with open(self._get_meta_path(), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        print("Knowledge base saved to disk successfully")
    
    def load_from_disk(self) -> bool:
        """Load the FAISS index and chunks from disk. Returns True if successful."""
        if not self._db_exists():
            return False
            
        try:
            # Load FAISS index
            self.index = faiss.read_index(self._get_index_path())
            print(f"Loaded FAISS index from {self._get_index_path()}")
            
            # Load chunks
            with open(self._get_chunks_path(), 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
            print(f"Loaded {len(self.chunks)} chunks from {self._get_chunks_path()}")
            
            self.is_initialized = True
            print("Knowledge base loaded from disk successfully")
            return True
            
        except Exception as e:
            print(f"Error loading knowledge base from disk: {e}")
            return False
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def chunk_text(self, text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> List[Dict]:
        """Split text into overlapping chunks for embedding."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            if len(chunk_text.strip()) > 50:  # Skip very short chunks
                chunks.append({
                    "text": chunk_text,
                    "source": source,
                    "chunk_id": len(chunks)
                })
        
        return chunks
    
    def build_index(self, save: bool = True) -> None:
        """Scan all PDFs in the knowledge base and build FAISS index."""
        print("Building knowledge base index...")
        all_chunks = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.docs_path):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_path = os.path.join(root, file)
                    relative_path = os.path.relpath(pdf_path, self.docs_path)
                    
                    text = self.extract_text_from_pdf(pdf_path)
                    if text:
                        chunks = self.chunk_text(text, relative_path)
                        all_chunks.extend(chunks)
                        print(f"  Processed: {relative_path} ({len(chunks)} chunks)")
        
        self.chunks = all_chunks
        
        if not self.chunks:
            print("Warning: No chunks extracted from knowledge base!")
            return
        
        # Generate embeddings
        print(f"Generating embeddings for {len(self.chunks)} chunks...")
        texts = [chunk["text"] for chunk in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype('float32'))
        
        self.is_initialized = True
        print(f"Knowledge base initialized with {len(self.chunks)} chunks")
        
        # Save to disk for future use
        if save:
            self.save_to_disk()
    
    def initialize(self) -> None:
        """Initialize the knowledge base - load from disk or build fresh."""
        print("Initializing knowledge base...")
        
        # Try to load from disk first
        if self.load_from_disk():
            print("Loaded existing knowledge base from database")
            return
        
        # Build fresh index if no saved database
        print("No existing database found. Building from PDFs...")
        self.build_index(save=True)
    
    def rebuild(self) -> None:
        """Force rebuild the knowledge base from PDFs."""
        print("Forcing rebuild of knowledge base...")
        self.build_index(save=True)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks using semantic similarity."""
        if not self.is_initialized:
            return []
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search FAISS index
        distances, indices = self.index.search(
            query_embedding.astype('float32'), 
            min(top_k, len(self.chunks))
        )
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(1 / (1 + distances[0][i]))  # Convert distance to similarity
                results.append(chunk)
        
        return results
    
    def get_context_for_review(self, submittal_type: str, description: str, specs: str) -> List[Dict]:
        """Get relevant context for reviewing a submittal."""
        # Combine submittal info into search query
        query = f"{submittal_type} {description} {specs}"
        return self.search(query, top_k=5)


# Global instance for reuse
_knowledge_base = None

def get_knowledge_base() -> KnowledgeBase:
    """Get or create the global knowledge base instance."""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBase()
        _knowledge_base.initialize()
    return _knowledge_base
