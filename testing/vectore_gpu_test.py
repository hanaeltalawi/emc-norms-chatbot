import torch
from langchain_huggingface import HuggingFaceEmbeddings

def test_embeddings_device():
    """Test if embeddings are actually using GPU"""
    print("EMBEDDINGS GPU USAGE TEST")
    print("=" * 40)
    
    # Test your exact configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"System reports device: {device}")
    
    try:
        # This is exactly what your VectorStoreManager does
        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': device}
        )
        
        print(f"✅ Embeddings initialized for device: {device}")
        
        # Test a small embedding
        test_text = "This is a test document for GPU verification"
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ Embedding generated successfully")
        print(f"   Embedding dimension: {len(embedding)}")
        print(f"   First 5 values: {embedding[:5]}")
        
        # Check where the model is actually running
        if hasattr(embeddings, 'client'):
            model_device = next(embeddings.client.parameters()).device
            print(f"   Model actually running on: {model_device}")
        else:
            print("   ⚠️  Cannot determine exact model device")
            
    except Exception as e:
        print(f"❌ Embeddings test failed: {e}")

if __name__ == "__main__":
    test_embeddings_device()