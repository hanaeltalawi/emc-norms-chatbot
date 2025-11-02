"""
Test script for Hybrid Search implementation

Usage: python testing/test_hybrid_search.py
"""

import sys
import os

# Add the parent directory to path to import your modules
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from hybrid_search import HybridSearch

def test_hybrid_search_fix():
    """Comprehensive test to verify the hybrid search is working correctly."""
    print("🚀 Testing Hybrid Search Implementation")
    print("=" * 60)
    
    # Initialize managers
    vector_manager = VectorStoreManager()
    document_processor = DocumentProcessor()
    
    # Use a test document (you might want to create a small test file)
    test_doc_path = "model_comparison_report/Translated_Norms.docx"
    
    if not os.path.exists(test_doc_path):
        print(f"❌ Test document not found: {test_doc_path}")
        print("Please create a small test DOCX file or use your main document")
        return None
    
    try:
        # Process test document
        print("📄 Processing test document...")
        text_content = document_processor.extract_text_and_tables_from_docx(test_doc_path)
        
        if not text_content:
            print("❌ Failed to extract text from test document")
            return None
        
        documents = document_processor.split_document_text(text_content)
        print(f"✅ Document split into {len(documents)} chunks")
        
        # Create vector store
        vectorstore = vector_manager.create_vectorstore(documents, "Test_Document")
        print("✅ Vector store created successfully")
        
        # Initialize hybrid search
        hybrid_searcher = HybridSearch(vectorstore)
        
        # Pre-load BM25 index
        all_docs = vectorstore.similarity_search("", k=len(documents) + 10)
        hybrid_searcher._build_bm25_index(all_docs)
        print("✅ Hybrid search initialized with BM25 index")
        
        # Test queries that should benefit from hybrid search
        test_queries = [
            "What are the peak limits for 5G n71?",  # Should have technical terms + semantic meaning
            "electromagnetic interference sources",  # Conceptual query
            "CISPR-25",  # Pure keyword match
            "test plan requirements",  # Semantic meaning
            "50 dBµV",  # Specific numeric value
        ]
        
        print(f"\n🔍 Testing {len(test_queries)} queries...")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: '{query}'")
            print("-" * 40)
            
            # Get explanation to see what's happening internally
            explanation = hybrid_searcher.search_with_explanation(query, k=3)
            
            if 'error' in explanation:
                print(f"❌ Error: {explanation['error']}")
                continue
            
            print(f"Alpha: {explanation['alpha']}")
            print(f"Semantic results: {explanation['semantic_results_count']}")
            print(f"Keyword results: {explanation['keyword_results_count']}")
            print(f"Combined unique results: {explanation['combined_results_count']}")
            
            print("\nTop 3 Results:")
            for j, detail in enumerate(explanation['details'][:3], 1):
                print(f"  {j}. Combined Score: {detail['combined_score']:.3f}")
                print(f"     Semantic: {detail['semantic_score']:.3f}, Keyword: {detail['keyword_score']:.3f}")
                print(f"     Source: {detail['source']}")
                content_preview = detail['document'].page_content.replace('\n', ' ')[:120] + "..."
                print(f"     Content: {content_preview}")
                print()
            
            # Also test the regular hybrid search
            hybrid_results = hybrid_searcher.hybrid_search(query, k=3)
            print(f"Hybrid Search returned {len(hybrid_results)} results")
        
        return hybrid_searcher
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_score_calculation():
    """Test the score normalization and combination logic specifically."""
    print("\n🧮 Testing Score Calculation Logic")
    print("=" * 50)
    
    # Create a mock hybrid search instance to test normalization
    class MockHybridSearch(HybridSearch):
        def __init__(self):
            # Bypass normal initialization for testing
            pass
        
        def test_normalization(self, scores):
            return self._normalize_scores(scores)
    
    mock_searcher = MockHybridSearch()
    
    # Test cases for score normalization
    test_cases = [
        ([1.0, 2.0, 3.0], "Normal scores"),
        ([1.0, 1.0, 1.0], "All equal scores"),
        ([0.1], "Single score"),
        ([], "Empty list"),
        ([5.0, 1.0, 3.0, 2.0], "Unsorted scores"),
    ]
    
    for scores, description in test_cases:
        print(f"\n{description}: {scores}")
        try:
            normalized = mock_searcher.test_normalization(scores)
            print(f"Normalized: {normalized}")
        except Exception as e:
            print(f"Error: {e}")

def test_edge_cases():
    """Test hybrid search with edge cases."""
    print("\n⚠️ Testing Edge Cases")
    print("=" * 50)
    
    # Initialize with a simple setup
    vector_manager = VectorStoreManager()
    document_processor = DocumentProcessor()
    
    # Create a minimal test document content
    minimal_content = """
    Test Document for EMC Analysis
    
    Section 1: Electromagnetic Compatibility
    EMC testing requires careful planning. The test plan must include frequency ranges and emission limits.
    
    Section 2: Test Parameters
    Frequency range: 30-1000 MHz
    Limit: 50 dBµV
    Standard: CISPR-25
    
    Table 1: Emission Limits
    Frequency | Peak Limit | Average Limit
    30-50 MHz | 60 dBµV | 50 dBµV
    50-100 MHz | 55 dBµV | 45 dBµV
    """
    
    try:
        # Create documents from minimal content
        doc = document_processor.split_document_text(minimal_content)
        vectorstore = vector_manager.create_vectorstore(doc, "Minimal_Test")
        
        hybrid_searcher = HybridSearch(vectorstore)
        all_docs = vectorstore.similarity_search("", k=len(doc) + 10)
        hybrid_searcher._build_bm25_index(all_docs)
        
        edge_case_queries = [
            "",  # Empty query
            "xyzabc123",  # No matches
            "a",  # Very short query
            "CISPR-25 AND 50 dBµV",  # Complex technical query
        ]
        
        for query in edge_case_queries:
            print(f"\nQuery: '{query}'")
            try:
                results = hybrid_searcher.hybrid_search(query, k=2)
                print(f"Results: {len(results)}")
                
                if results:
                    explanation = hybrid_searcher.search_with_explanation(query, k=2)
                    print(f"Sources: {[detail['source'] for detail in explanation['details']]}")
                else:
                    print("No results found (expected for edge cases)")
                    
            except Exception as e:
                print(f"Error: {e}")
                
    except Exception as e:
        print(f"Error in edge case testing: {e}")

def performance_test(hybrid_searcher):
    """Test the performance of hybrid search vs semantic-only search."""
    if not hybrid_searcher:
        print("❌ No hybrid searcher available for performance test")
        return
    
    print("\n⚡ Performance Comparison: Hybrid vs Semantic Search")
    print("=" * 60)
    
    import time
    
    test_queries = [
        "emission limits",
        "test plan",
        "CISPR-25",
        "electromagnetic compatibility",
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        
        # Test hybrid search
        start_time = time.time()
        hybrid_results = hybrid_searcher.hybrid_search(query, k=5)
        hybrid_time = time.time() - start_time
        
        # Test semantic-only search (fallback)
        start_time = time.time()
        semantic_results = hybrid_searcher.vectorstore.similarity_search(query, k=5)
        semantic_time = time.time() - start_time
        
        print(f"Hybrid search: {len(hybrid_results)} results, {hybrid_time:.3f}s")
        print(f"Semantic search: {len(semantic_results)} results, {semantic_time:.3f}s")
        print(f"Time difference: {hybrid_time - semantic_time:+.3f}s")
        
        # Check if results are different
        hybrid_content = [doc.page_content[:50] for doc in hybrid_results]
        semantic_content = [doc.page_content[:50] for doc in semantic_results]
        
        if hybrid_content != semantic_content:
            print("✅ Results are DIFFERENT - hybrid search is working!")
        else:
            print("⚠️  Results are similar")

if __name__ == "__main__":
    print("🧪 Hybrid Search Test Suite")
    print("This test verifies the corrected hybrid search implementation.")
    print()
    
    # Run all tests
    searcher = test_hybrid_search_fix()
    
    if searcher:
        test_score_calculation()
        test_edge_cases()
        performance_test(searcher)
    
    print("\n" + "=" * 60)
    print("🎯 Test Suite Complete")
    print("Check the output above to verify:")
    print("✅ Scores are properly normalized (0-1 range)")
    print("✅ Both semantic and keyword results are combined")
    print("✅ Different query types work (technical, conceptual, keyword)")
    print("✅ Edge cases are handled gracefully")
    print("✅ Hybrid search provides different results than semantic-only")