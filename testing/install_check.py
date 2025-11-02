import importlib
import pkg_resources

def check_installations():
    """Check if required packages are installed with versions"""
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers', 
        'langchain': 'LangChain',
        'langchain_huggingface': 'LangChain HuggingFace',
        'streamlit': 'Streamlit',
        'docx': 'Python DOCX',
        'rank_bm25': 'BM25',
        'chromadb': 'ChromaDB'
    }
    
    print("PACKAGE INSTALLATION CHECK")
    print("=" * 40)
    
    for pkg, name in packages.items():
        try:
            if pkg == 'torch':
                import torch
                version = torch.__version__
                cuda_status = " (CUDA)" if torch.cuda.is_available() else " (CPU)"
                print(f"✅ {name}: {version}{cuda_status}")
            else:
                spec = importlib.util.find_spec(pkg)
                if spec is not None:
                    version = pkg_resources.get_distribution(pkg).version
                    print(f"✅ {name}: {version}")
                else:
                    print(f"❌ {name}: NOT FOUND")
        except Exception as e:
            print(f"❌ {name}: Error - {e}")

if __name__ == "__main__":
    check_installations()