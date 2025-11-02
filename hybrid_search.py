from typing import List, Dict, Any, Tuple
from langchain.schema import Document
from rank_bm25 import BM25Okapi
import numpy as np
import re

class HybridSearch:
    """Combines semantic and keyword search for better retrieval."""
    
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.bm25_index = None
        self.documents = None
        
    def _build_bm25_index(self, documents: List[Document]):
        """Build BM25 keyword search index with better tokenization."""
        tokenized_docs = []
        
        for doc in documents:
            text = doc.page_content.lower()
            
            # Enhanced tokenization for technical documents
            tokens = re.findall(r'''
                [a-zA-Z]{2,}          # Words with 2+ letters
                | \d+                  # Numbers
                | [a-zA-Z]+-\w+        # Hyphenated terms
                | \w+\.\w+            # Terms with dots (e.g., "e.m.c.")
                | [A-Z]{2,}           # Acronyms (all caps)
            ''', text, re.VERBOSE)
            
            # Add metadata context if available
            if 'type' in doc.metadata and doc.metadata['type'] == 'table':
                tokens.extend(['table', 'data', 'row', 'column', 'header'])
            
            # Remove very short tokens but keep numbers
            tokens = [token for token in tokens if len(token) > 1 or token.isdigit()]
            
            tokenized_docs.append(tokens)
        
        self.bm25_index = BM25Okapi(tokenized_docs)
        self.documents = documents

    def _get_all_chunks_from_document(self) -> List[Document]:
        """Get ALL chunks from the single document."""
        try:
            # Since we only have one document, get all chunks using empty query
            return self.vectorstore.similarity_search("", k=1000)
        except Exception as e:
            print(f"Error getting document chunks: {e}")
            return []
    
    def _normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to 0-1 range using min-max scaling."""
        if not scores:
            return []
        
        scores_array = np.array(scores)
        if np.max(scores_array) == np.min(scores_array):
            # All scores are equal, return uniform scores
            return [0.5] * len(scores)
        
        return (scores_array - np.min(scores_array)) / (np.max(scores_array) - np.min(scores_array))
    
    def _get_semantic_results(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Get semantic search results with normalized scores."""
        try:
            # Get results with relevance scores
            semantic_results = self.vectorstore.similarity_search_with_relevance_scores(query, k=k*2)
            
            if not semantic_results:
                return []
            
            # Extract and normalize scores
            documents = [doc for doc, score in semantic_results]
            raw_scores = [score for doc, score in semantic_results]
            normalized_scores = self._normalize_scores(raw_scores)
            
            return list(zip(documents, normalized_scores))
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []
    
    def _get_keyword_results(self, query: str, k: int) -> List[Tuple[Document, float]]:
        """Get keyword search results with normalized BM25 scores."""
        if self.bm25_index is None:
            return []
        
        try:
            # Tokenize query
            query_tokens = re.findall(r'\w+', query.lower()) if query else []
            
            if not query_tokens:
                return []
            
            # Get BM25 scores for all documents
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            if hasattr(bm25_scores, 'tolist'):
                bm25_scores = bm25_scores.tolist()
            
            # Create document-score pairs
            doc_score_pairs = list(zip(self.documents, bm25_scores))
            
            # Sort by BM25 score (descending) and take top k
            doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
            top_pairs = doc_score_pairs[:k*2]
            
            if not top_pairs:
                return []
            
            # Extract and normalize scores
            documents = [doc for doc, score in top_pairs]
            raw_scores = [score for doc, score in top_pairs]
            normalized_scores = self._normalize_scores(raw_scores)
            
            return list(zip(documents, normalized_scores))
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
    
    def hybrid_search(self, query: str, k: int = 10, alpha: float = 0.6) -> List[Document]:
        """
        Perform proper hybrid search combining semantic and keyword search.
        
        Args:
            query: The search query
            k: Number of results to return
            alpha: Weight for semantic vs keyword (0.6 = 60% semantic, 40% keyword)
        
        Returns:
            List of documents ranked by hybrid score
        """
        try:
            # Build BM25 index if needed
            if self.bm25_index is None:
                document_chunks = self._get_all_chunks_from_document()
                if document_chunks:
                    self._build_bm25_index(document_chunks)
                else:
                    # Fallback to semantic only
                    return self.vectorstore.similarity_search(query, k=k)
            
            # Get results from both systems independently
            semantic_results = self._get_semantic_results(query, k)
            keyword_results = self._get_keyword_results(query, k)
            
            # If one system fails, fallback to the other
            if not semantic_results and not keyword_results:
                return []
            elif not semantic_results:
                return [doc for doc, score in keyword_results[:k]]
            elif not keyword_results:
                return [doc for doc, score in semantic_results[:k]]
            
            # Create a combined dictionary of all unique documents
            combined_scores = {}
            document_source = {}  # Track which system found the document
            
            # Add semantic results
            for doc, score in semantic_results:
                doc_id = hash(doc.page_content)
                combined_scores[doc_id] = alpha * score
                document_source[doc_id] = doc
            
            # Add keyword results (accumulate scores)
            for doc, score in keyword_results:
                doc_id = hash(doc.page_content)
                if doc_id in combined_scores:
                    # Document found by both systems - combine scores
                    combined_scores[doc_id] += (1 - alpha) * score
                else:
                    # Document only found by keyword system
                    combined_scores[doc_id] = (1 - alpha) * score
                    document_source[doc_id] = doc
            
            # Convert to list and sort by combined score
            final_results = []
            for doc_id, score in combined_scores.items():
                final_results.append((document_source[doc_id], score))
            
            # Sort by combined score (descending)
            final_results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top k results
            return [doc for doc, score in final_results[:k]]
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            # Fallback to simple semantic search
            try:
                return self.vectorstore.similarity_search(query, k=k)
            except:
                return []
    
    def search_with_explanation(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Debug method to show search scoring details."""
        try:
            # Build BM25 index if not exists
            if self.bm25_index is None:
                document_chunks = self._get_all_chunks_from_document()
                if document_chunks:
                    self._build_bm25_index(document_chunks)
            
            # Get results from both systems
            semantic_results = self._get_semantic_results(query, k*2)
            keyword_results = self._get_keyword_results(query, k*2)
            
            # Combine using the same logic as hybrid_search
            combined_scores = {}
            document_source = {}
            explanation_data = {}
            
            alpha = 0.6  # Use the same alpha
            
            # Process semantic results
            for doc, score in semantic_results:
                doc_id = hash(doc.page_content)
                explanation_data[doc_id] = {
                    'document': doc,
                    'semantic_score': score,
                    'keyword_score': 0,
                    'combined_score': alpha * score,
                    'source': 'semantic_only'
                }
                document_source[doc_id] = doc
            
            # Process keyword results
            for doc, score in keyword_results:
                doc_id = hash(doc.page_content)
                if doc_id in explanation_data:
                    # Document found by both systems
                    explanation_data[doc_id]['keyword_score'] = score
                    explanation_data[doc_id]['combined_score'] = (
                        alpha * explanation_data[doc_id]['semantic_score'] + 
                        (1 - alpha) * score
                    )
                    explanation_data[doc_id]['source'] = 'both'
                else:
                    # Document only found by keyword
                    explanation_data[doc_id] = {
                        'document': doc,
                        'semantic_score': 0,
                        'keyword_score': score,
                        'combined_score': (1 - alpha) * score,
                        'source': 'keyword_only'
                    }
                    document_source[doc_id] = doc
            
            # Convert to sorted list
            detailed_results = list(explanation_data.values())
            detailed_results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            return {
                'query': query,
                'alpha': alpha,
                'semantic_results_count': len(semantic_results),
                'keyword_results_count': len(keyword_results),
                'combined_results_count': len(detailed_results),
                'details': detailed_results[:k]
            }
            
        except Exception as e:
            print(f"Error in search explanation: {e}")
            return {'query': query, 'details': [], 'error': str(e)}
        
