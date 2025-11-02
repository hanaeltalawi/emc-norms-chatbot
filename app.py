import streamlit as st
import tempfile
import os
import re
import tempfile
import time
import atexit
import gc

from document_processor import DocumentProcessor
from vector_store_manager import VectorStoreManager
from hybrid_search import HybridSearch
from model_manager import ModelManager, MODELS
from query_filter import QueryFilter

# AUTO-CLEANUP ON REFRESH/RELOAD
if 'app_started' not in st.session_state:
    # First time app runs or refresh - clean everything
    st.session_state.vectorstore = None
    st.session_state.document_chunk_count = 0
    st.session_state.document_processed = False
    st.session_state.auto_processed = False
    st.session_state.document_uploaded = False
    st.session_state.document_name = None
    st.session_state.chat_history = []
    st.session_state.last_uploaded_name = None
    st.session_state.app_started = True
    print("🔄 App fresh start - session cleaned")
else:
    print("🔁 App reloaded - session state preserved")

# Initialize managers
vector_manager = VectorStoreManager()
document_processor = DocumentProcessor()
model_manager = ModelManager()

# PRE-LOAD MODELS IMMEDIATELY
if 'models_loaded' not in st.session_state:
    with st.spinner("🚀 Loading AI models..."):
        try:
            # Pre-load only top 2-3 most used models
            model_manager.create_ollama_llm("phi3:latest")
            model_manager.create_ollama_llm("mistral:7b")
            print("✅ Models pre-loaded successfully")
            st.session_state.models_loaded = True
        except Exception as e:
            print(f"❌ Model pre-loading failed: {e}")
            st.session_state.models_loaded = False



# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #1f77b4;
    }
    .success-card {
        border-left: 4px solid #28a745;
        background-color: #d4edda;
    }
    .warning-card {
        border-left: 4px solid #ffc107;
        background-color: #fff3cd;
    }
    .model-card {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #bee5eb;
    }
    .metric-card {
        text-align: center;
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        margin: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit App Configuration
st.set_page_config(
    page_title="EMC Norms Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main Header
st.markdown('<h1 class="main-header">🤖 EMC Norms Chatbot</h1>', unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <p style="font-size: 1.2rem; color: #666;">Upload technical documents and compare AI model performance</p>
</div>
""", unsafe_allow_html=True)

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'document_chunk_count' not in st.session_state:       
    st.session_state.document_chunk_count = 0
if 'document_name' not in st.session_state:
    st.session_state.document_name = None
if 'document_processed' not in st.session_state:
    st.session_state.document_processed = False
if 'auto_processed' not in st.session_state:
    st.session_state.auto_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'document_uploaded' not in st.session_state:
    st.session_state.document_uploaded = False

# Add cleanup function
def cleanup_session():
    """Clean up document resources while keeping models loaded."""
    try:
        # Clear vectorstore properly
        if st.session_state.vectorstore and hasattr(st.session_state.vectorstore, 'delete_collection'):
            try:
                st.session_state.vectorstore.delete_collection()
            except:
                pass
        
        # Reset ALL document state
        st.session_state.vectorstore = None
        st.session_state.document_chunk_count = 0
        st.session_state.document_processed = False
        st.session_state.auto_processed = False
        st.session_state.document_uploaded = False
        st.session_state.document_name = None
        st.session_state.chat_history = []
        st.session_state.last_uploaded_name = None
        
        # ONLY clear retrieval chains, not models
        model_manager.clear_retrieval_chain_cache("document_change")
        
        gc.collect()
        print("✅ Document cleared (models kept loaded)")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")

def cleanup_everything():
    """Comprehensive cleanup function"""
    try:
        if st.session_state.vectorstore and hasattr(st.session_state.vectorstore, 'delete_collection'):
            try:
                st.session_state.vectorstore.delete_collection()
            except:
                pass
        st.session_state.vectorstore = None
        gc.collect()
    except:
        pass

# Register the cleanup
atexit.register(cleanup_everything)

# Register cleanup
atexit.register(cleanup_session)

# Sidebar for settings
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    # Model Selection in an expander
    with st.expander("🤖 AI Models", expanded=True):
        model_options = list(MODELS.keys())
        selected_model = st.selectbox(
            "Select Model",
            model_options,
            format_func=lambda x: MODELS[x]['name']
        )
        
        compare_mode = st.checkbox("Model Comparison", value=False)
        
        if compare_mode:
            comparison_models = st.multiselect(
                "Models to Compare",
                model_options,
                default=[selected_model],
                format_func=lambda x: MODELS[x]['name']
            )
    
    # Storage settings
    with st.expander("💾 Storage", expanded=True):
        st.info("Real-time mode only - documents exist only during this session")

    # Display system info
    st.markdown("### 📊 System Status")
    device = 'GPU (CUDA)' if vector_manager.embeddings.model_kwargs.get('device') == 'cuda' else 'CPU'
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Device", device)
    with col2:
        st.metric("Available Models", len(MODELS))

    # Current document status
    st.markdown("### 📄 Current Document")
    if st.session_state.document_uploaded and st.session_state.vectorstore:
        document_name = st.session_state.document_name or "Uploaded Document"
        chunk_count = st.session_state.document_chunk_count

        st.markdown(f'<div class="card success-card">'
                f'<strong>✅ {document_name}</strong>'
                f'</div>', unsafe_allow_html=True)
        st.metric("Document Chunks", chunk_count)
    else:
        st.info("No document loaded")

    # Cache management
    st.markdown("---")
    st.markdown("### 🔧 Cache Management")
    if st.button("🔄 Clear Uploaded Document"):
        cleanup_session()
        st.success("✅ Document cleared! You can now upload a new file.")
        st.rerun()

# Main content area
tab1, tab2 = st.tabs(["📁 Document Management", "💬 Chatbot"])

with tab1:
    st.markdown("### 📁 Document Management")
    
    # Document upload section
    with st.container():
        st.markdown("#### 📤 Upload New Document")

        # Check if a document has already been uploaded
        if st.session_state.document_uploaded:
            st.warning("⚠️ You can only upload one document per session. Please clear the current document to upload a new one.")
            uploaded_file = None
        else:
            uploaded_file = st.file_uploader(
                "Drag and drop or click to upload a DOCX file", 
                type=['docx'],
                key="file_uploader"
            )

        # Detect file change and auto-cleanup
        if "last_uploaded_name" not in st.session_state:
            st.session_state.last_uploaded_name = None
        
        if uploaded_file is not None and not st.session_state.document_uploaded:            
            # Check for file change and auto-cleanup
            if (st.session_state.last_uploaded_name and 
                st.session_state.last_uploaded_name != uploaded_file.name and
                st.session_state.document_uploaded):
                
                # Different file detected - auto-cleanup previous document
                st.info("🔄 New file detected - cleaning previous document...")
                cleanup_session()  # Call your existing cleanup function
                st.rerun()

            if st.session_state.last_uploaded_name != uploaded_file.name:
                # New file selected → reset any prior processing state
                st.session_state.auto_processed = False
                st.session_state.document_processed = False
                st.session_state.vectorstore = None
                st.session_state.document_name = None
                # Drop any stale name input state if it existed
                st.session_state.pop("doc_name_input", None)
                st.session_state.last_uploaded_name = uploaded_file.name

            if not st.session_state.auto_processed:
                original_name = uploaded_file.name
                doc_name_without_ext = os.path.splitext(original_name)[0]

                # Fixed, non-editable reference name
                document_name = doc_name_without_ext
                st.caption(f"Document name (reference): **{document_name}**")

                # Optional: sanitize the name to keep folders/collections safe
                document_name = re.sub(r"[^-\w\s.]+", "_", document_name).strip()

                # Auto-process immediately
                with st.spinner("🔄 Processing document..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_file_path = tmp_file.name

                        text_content = document_processor.extract_text_and_tables_from_docx(tmp_file_path)
                        os.unlink(tmp_file_path)

                        if text_content:
                            final_documents = document_processor.split_document_text(text_content)

                            # Store the actual count of chunks
                            chunk_count = len(final_documents)
                            st.session_state.document_chunk_count = chunk_count

                            # Create in-memory vectorstore
                            vectorstore = vector_manager.create_vectorstore(final_documents, document_name)

                            if vectorstore:
                                # Pre-load hybrid search index immediately
                                hybrid_searcher = HybridSearch(vectorstore)
                                all_docs = vectorstore.similarity_search("", k=chunk_count + 10)
                                hybrid_searcher._build_bm25_index(all_docs)

                                # Calculate and store chunk count
                                chunk_count = len(final_documents)

                                # CLEAR any existing document first
                                if st.session_state.vectorstore and hasattr(st.session_state.vectorstore, 'delete_collection'):
                                    try:
                                        st.session_state.vectorstore.delete_collection()
                                    except:
                                        pass
                                
                                # Store the new document
                                st.session_state.vectorstore = vectorstore
                                st.session_state.document_chunk_count = chunk_count
                                st.session_state.document_name = document_name
                                
                                st.session_state.auto_processed = True
                                st.session_state.document_processed = True
                                st.session_state.document_uploaded = True
                                
                                # Clear retrieval cache when new document is uploaded
                                model_manager.clear_retrieval_chain_cache("document_change")
                                st.session_state.auto_processed = True
                                st.session_state.document_processed = True
                                st.session_state.document_uploaded = True
                                
                                st.success("✅ Document processed successfully!")
                                st.rerun()
                        else:
                            st.error("❌ Failed to extract text from document")

                    except Exception as e:
                        st.error(f"❌ Error processing document: {str(e)}")

        # Reset auto-processed flag when file changes
        if uploaded_file is None and st.session_state.get("auto_processed"):
            st.session_state.auto_processed = False

    st.markdown("📊 Current Document")
    if st.session_state.document_uploaded and st.session_state.vectorstore:
        document_name = st.session_state.document_name or "Uploaded Document"
        chunk_count = st.session_state.document_chunk_count

        st.markdown(f'<div class="card success-card">'
                f'<strong>✅ {document_name}</strong>'
                f'</div>', unsafe_allow_html=True)
        st.metric("Document Chunks", chunk_count)
    else:
        st.info("No document loaded.")

              
with tab2:
    st.markdown("### 💬 Document Analysis")
    
    if not st.session_state.document_processed or not st.session_state.vectorstore:
        st.info("👆 Upload a document in the Document Management tab to start analyzing")
    else:
        document_name = st.session_state.document_name or "Unknown Document"
        st.markdown(f"**Analyzing:** {document_name}")
        
        # Chat Container
        chat_container = st.container()
        
        # Display chat history in the container
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(message["content"])
                else:   # assitant
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(message["content"])
                        if "response_time" in message:
                            st.caption(f"⏱️ {message['response_time']:.2f}s • {message.get('model', 'AI')}")
        
        #Input at the bottom (outside container)
        if st.session_state.vectorstore:
            document_name = st.session_state.document_name or "the document"
            user_query = st.chat_input(
                f"Ask a question about {document_name}...",
                key="chat_input"
            )
        else:
            user_query = st.chat_input(
                "Upload a document to start asking questions...",
                key="chat_input"
            )

        # Process new query
        if user_query:
            # Add user message to history and display immediately
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            
            # Immediately show user message
            with chat_container:
                with st.chat_message("user", avatar="👤"):
                    st.markdown(user_query)
            
            if not QueryFilter.is_document_related_query(user_query):
                # Show AI response for non-document questions
                with chat_container:
                    with st.chat_message("assistant", avatar="🤖"):
                        st.info("I specialize in answering questions about the uploaded document. Please ask about its content.")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": "I specialize in answering questions about the uploaded document. Please ask about its content."
                    })
            else:
                # Show loading indicator
                with chat_container:
                    with st.chat_message("assistant", avatar="🤖"):
                        thinking_placeholder = st.empty()
                        thinking_placeholder.info("🤔 Thinking...")

                try:
                    # Get the CURRENT active vectorstore
                    current_vectorstore = st.session_state.vectorstore
                    # Create hybrid searcher and pre-load with known chunks
                    hybrid_searcher = HybridSearch(current_vectorstore)
                    
                    if current_vectorstore is None:
                        thinking_placeholder.empty()
                        with chat_container:
                            with st.chat_message("assistant", avatar="🤖"):
                                st.error("❌ No document is currently loaded. Please upload a document first.")
                        
                        st.session_state.chat_history.append({
                            "role": "assistant", 
                            "content": "❌ No document is currently loaded. Please upload a document first."
                        })
                    else:
                        if compare_mode and comparison_models:
                            # Model comparison
                            comparison_results = []
                            thinking_placeholder.empty()

                            with chat_container:
                                with st.chat_message("assistant", avatar="🤖"):
                                    st.markdown("### 🔄 Model Comparison Results")

                                    for model_name in comparison_models:
                                        with st.spinner(f"Analyzing with {MODELS[model_name]['name']}..."):
                                            try:
                                                retrieval_chain = model_manager.create_retrieval_chain(model_name, current_vectorstore)
                                                
                                                start_time = time.time()
                                                # Pass the current vectorstore to the retrieval chain
                                                response = retrieval_chain({"input": user_query}, vectorstore=current_vectorstore)
                                                end_time = time.time()
                                                
                                                response_time = end_time - start_time
                                                
                                                with st.expander(f"{MODELS[model_name]['name']} ({response_time:.2f}s)", expanded=False):
                                                    st.markdown(response['answer'])
                                                    st.caption(f"Response time: {response_time:.2f}s")
                                                
                                                comparison_results.append({
                                                    "model": MODELS[model_name]['name'],
                                                    "response": response['answer'],
                                                    "time": response_time
                                                })
                                            except Exception as e:
                                                st.error(f"Error with {model_name}: {str(e)}")
                            
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": "Generated comparative analysis across multiple models",
                                "comparison": comparison_results
                            })
                            
                        else:
                            # Single model response
                            retrieval_chain = model_manager.create_retrieval_chain(selected_model, current_vectorstore)
                            
                            start_time = time.time()
                            # Pass the current vectorstore to the retrieval chain
                            response = retrieval_chain({"input": user_query}, vectorstore=current_vectorstore)
                            end_time = time.time()
                            
                            response_time = end_time - start_time
                            
                            # Update the UI with the actual response
                            thinking_placeholder.empty()

                            with chat_container:
                                with st.chat_message("assistant", avatar="🤖"):
                                    st.markdown(response['answer'])
                                    st.caption(f"⏱️ {response_time:.2f}s • {MODELS[selected_model]['name']}")
                                    
                                    # Show context sources in expander
                                    with st.expander("📚 Source References"):
                                        if 'context' in response and response['context']:
                                            for i, doc in enumerate(response['context'], 1):
                                                st.markdown(f"**Context {i}:**")
                                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                                if i < len(response['context']):
                                                    st.markdown("---")
                            
                            st.session_state.chat_history.append({
                                "role": "assistant", 
                                "content": response['answer'],
                                "response_time": response_time,
                                "model": MODELS[selected_model]['name']
                            })

                    # Pre-load the BM25 index with the actual document chunks we already have
                    if st.session_state.document_chunk_count and not hybrid_searcher.bm25_index:
                        chunk_count = st.session_state.document_chunk_count
                        
                        # Get all documents using the actual chunk count we know
                        all_docs = current_vectorstore.similarity_search("", k=chunk_count + 10)
                        hybrid_searcher._build_bm25_index(all_docs)

                except Exception as e:
                    thinking_placeholder.empty()
                    with chat_container:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.error(f"❌ Error generating response: {str(e)}")
                    
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"Error: {str(e)}"
                    })

        st.markdown("""
        <style>
            .stChatMessage {
                padding: 1rem;
                border-radius: 15px;
                margin-bottom: 1rem;
            }
            [data-testid="stChatMessage"] {
                max-width: 80%;
            }
            [data-testid="stChatMessage"][aria-label="A chat message from user."] {
                margin-left: auto;
                background-color: #1f77b4;
                color: white;
            }
            [data-testid="stChatMessage"][aria-label="A chat message from assistant."] {
                margin-right: auto;
                background-color: #f0f2f6;
            }
            .stChatInput {
                position: fixed;
                bottom: 20px;
                left: 60%;
                transform: translateX(-50%);
                width: 85% !important;
                max-width: 800px;
                background: white;
                z-index: 1000;
                border: 1px solid #e6e6e6;
                border-radius: 20px;
                box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
                padding: 8px 16px;
            }       

        </style>
        """, unsafe_allow_html=True)