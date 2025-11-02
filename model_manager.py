from langchain_community.llms import Ollama
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
from typing import Dict, Any
import streamlit as st
import gc

from hybrid_search import HybridSearch


# Model list with better configurations
MODELS = {
    "phi3:latest": {
        "name": "Phi-3 3.8B",
        "temperature": 0.2,
        "top_p": 0.8,
        "num_predict": 256,
        "description": "Microsoft's efficient model"
    },
    "tinyllama:latest": {
        "name": "TinyLlama 1.1B", 
        "temperature": 0.3,
        "top_p": 0.7,
        "num_predict": 128,
        "description": "Ultra-fast tiny model"
    },
    "phi:latest": {
        "name": "Phi-2 2.7B", 
        "temperature": 0.2,
        "top_p": 0.8,
        "num_predict": 256,
        "description": "Balanced performance"
    },
    "mistral:7b": {
        "name": "Mistral 7B",
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 384,
        "description": "High-quality 7B model"
    },
    "llama3.1:8b": {
        "name": "Llama 3.1 8B",
        "temperature": 0.1,
        "top_p": 0.9,
        "num_predict": 512,
        "description": "Large, high-quality model"
    }
}

class ModelManager:
    """Manages LLM model initialization and chain creation."""
    
    @st.cache_resource(show_spinner=False, max_entries=5)
    def create_ollama_llm(_self, model_name: str):
        """Create an Ollama LLM instance with caching - one per model."""
        try:
            model_config = MODELS.get(model_name, {})
            print(f"Initializing and caching LLM: {model_name}")
            
            return Ollama(
                model=model_name,
                base_url="http://localhost:11434",
                temperature=model_config.get('temperature', 0.1),
                top_p=model_config.get('top_p', 0.9),
                num_predict=model_config.get('num_predict', 256),
                stop=["Human:", "AI:", "Question:", "Context:"]
            )
        except Exception as e:
            raise Exception(f"Error initializing Ollama LLM: {e}")
        
    @staticmethod
    @st.cache_data(show_spinner=False, max_entries=1)
    def create_prompt() -> ChatPromptTemplate:
        """Create prompt template for better responses."""
        return ChatPromptTemplate.from_template(
            """You are a technical document expert. Answer questions based ONLY on the provided document context.
            
CONTEXT FROM DOCUMENT: {context}

QUESTION: {input}

INSTRUCTIONS:
1. Answer ONLY based on the information in the context above
2. If the answer is not in the context, say "I cannot find this information in the provided document"
3. Be precise and cite specific details, tables, or sections from the document
4. For technical terms, use the exact terminology from the context
5. If referring to tables, describe the table structure and data clearly
6. Keep answers concise but complete
7. Do not repeat the question in your answer

ANSWER:"""
        )
        
    @st.cache_resource(show_spinner=False, max_entries=20)
    def create_retrieval_chain(_self, model_name: str, _vectorstore):
        """Create a retrieval chain with caching for question answering"""
        try:
            cache_key = f"{model_name}_retrieval_chain"

            print(f"Creating retrieval chain for model: {model_name}")
            llm = _self.create_ollama_llm(model_name)
            prompt = ModelManager.create_prompt()
            
            document_chain = create_stuff_documents_chain(llm, prompt)

            def retrieval_chain_func(inputs: Dict[str, Any], vectorstore=None) -> Dict[str, Any]:
                """Custom retrieval chain using hybrid search."""
                # Use the vectorstore passed as parameter, not the cached one
                if vectorstore is None:
                    return {
                        "answer": "No document context available. Please upload a document first.",
                        "context": []
                    }
                
                query = inputs["input"]
                
                # Create hybrid searcher with current vectorstore
                hybrid_searcher = HybridSearch(vectorstore)
                
                # Get context using hybrid search
                context_docs = hybrid_searcher.hybrid_search(query, k=5, alpha=0.6)
                
                # Make sure we have valid documents
                if not context_docs:
                    return {
                        "answer": "I cannot find relevant information in the document to answer this question.",
                        "context": []
                    }
                
                # Invoke the document chain
                result = document_chain.invoke({
                    "context": context_docs, 
                    "input": query
                })
                
                # Ensure we return a proper response format with context
                if isinstance(result, str):
                    return {
                        "answer": result,
                        "context": context_docs
                    }
                elif isinstance(result, dict) and "answer" in result:
                    return {
                        "answer": result["answer"],
                        "context": context_docs
                    }
                else:
                    return {
                        "answer": str(result),
                        "context": context_docs
                    }
            
            return retrieval_chain_func
            
        except Exception as e:
            raise Exception(f"Error creating retrieval chain: {e}")
        
    def clear_retrieval_chain_cache(self, reason: str = "manual"):
        """Clear only retrieval chain caches (keeps models loaded)."""
        try:
            # Only clear retrieval chains, NOT the loaded models
            if hasattr(st, 'cache_resource'):
                self.create_retrieval_chain.clear()  # Document chains only

            print(f"Cleared retrieval chains (models kept loaded): {reason}")
            
        except Exception as e:
            print(f"Warning: Error clearing retrieval caches: {e}")