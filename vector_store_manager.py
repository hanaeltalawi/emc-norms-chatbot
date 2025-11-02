import gc
from typing import Optional
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import torch
import streamlit as st

class VectorStoreManager:
    """Manages all interactions with the Chroma vector database."""
    
    def __init__(self):
        # Initialize with cached resources
        self._embeddings = self._load_embedding_model()

    @st.cache_resource(show_spinner=False, max_entries=1)
    def _load_embedding_model(_self):
        """Load the embedding model with caching - only once per session."""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading embedding model on {device}...")
        return HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': device}
        )
    
    @property
    def embeddings(self):
        return self._embeddings
        
    def create_vectorstore(self, documents: list[Document], document_name: str) -> Optional[Chroma]:
        """Create a new in-memory vectorstore from documents."""
        try:
            # Create in-memory vectorstore (no persistence)
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            return vectorstore
                
        except Exception as e:
            raise Exception(f"Error creating vectorstore: {e}")

    @st.cache_resource(show_spinner="Loading document...", max_entries=10)
    def load_vectorstore(_self, document_name: str) -> Optional[Chroma]:
        """This method is no longer needed for real-time mode but kept for compatibility."""
        # For real-time mode, we don't load from disk
        return None
    
    def clear_vectorstore_cache(self, document_name: str):
        """Clear specific vectorstore cache for a document."""
        try:            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Error clearing vectorstore cache: {e}")