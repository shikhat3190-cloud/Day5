# !pip install -q google-generativeai chromadb

# End-to-End RAG System with Gemini Flash and ChromaDB

import os
import re
from typing import List, Dict, Tuple
import google.generativeai as genai
import chromadb
from chromadb.config import Settings

# ==================== CONFIGURATION ====================

class RAGConfig:
    """Configuration for RAG system"""
    GEMINI_MODEL = "gemini-2.5-flash-lite"
    EMBEDDING_MODEL = "models/text-embedding-004"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50 # Chunk1 and Chunk 2 will have 50 char similar 
    TOP_K_RESULTS = 3
    COLLECTION_NAME = "rag_documents"

from google.colab import userdata
GEMINI_API_KEY = userdata.get("GOOGLE_API_KEY")
# Set your API key  # Replace with your actual API key
genai.configure(api_key=GEMINI_API_KEY)


# ==================== CUSTOM TEXT LOADER ====================

class CustomTextLoader:
    """Custom loader for reading .txt files"""

    @staticmethod
    def load_file(file_path: str) -> str:
        """Load text from a single .txt file"""
        if not file_path.endswith('.txt'):
            raise ValueError("Only .txt files are supported")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        return content

    @staticmethod
    def load_directory(directory_path: str) -> List[Dict[str, str]]:
        """Load all .txt files from a directory"""
        documents = []

        for filename in os.listdir(directory_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(directory_path, filename)
                content = CustomTextLoader.load_file(file_path)
                documents.append({
                    'content': content,
                    'source': filename,
                    'path': file_path
                })

        return documents



# ==================== CUSTOM TEXT SPLITTER ====================

class CustomTextSplitter:
    """Custom text splitter with overlap support"""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks with overlap"""
        # Clean the text
        text = self._clean_text(text)

        # Split by sentences first for better semantic boundaries
        sentences = self._split_into_sentences(text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(' '.join(current_chunk))

                # Create overlap by keeping last few sentences
                overlap_text = ' '.join(current_chunk)
                overlap_sentences = []
                overlap_length = 0

                for s in reversed(current_chunk):
                    if overlap_length + len(s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, s)
                        overlap_length += len(s)
                    else:
                        break

                current_chunk = overlap_sentences
                current_length = overlap_length

            current_chunk.append(sentence)
            current_length += sentence_length

        # Add the last chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\-\'\"()]', '', text)
        return text.strip()

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitter
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


# ==================== EMBEDDING MANAGER ====================

class EmbeddingManager:
    """Manage embeddings using Gemini's embedding model"""

    def __init__(self, model_name: str = RAGConfig.EMBEDDING_MODEL):
        self.model_name = model_name

    def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text"""
        result = genai.embed_content(
            model=self.model_name,
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query"""
        result = genai.embed_content(
            model=self.model_name,
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.embed_text(text)
            embeddings.append(embedding)
        return embeddings
