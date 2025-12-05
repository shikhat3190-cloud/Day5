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

