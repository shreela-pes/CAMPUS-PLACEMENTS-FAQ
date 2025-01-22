import streamlit as st
import nest_asyncio
import logging
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.fastembed import FastEmbedEmbedding
from llama_index.llms.groq import Groq
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.postprocessor import LLMRerank
from llama_index.core.query_engine import RetryQueryEngine
from llama_index.core.callbacks import CallbackManager
import pickle
from pathlib import Path
import time

# Page config
st.set_page_config(page_title="Campus Placement RAG Chat", layout="wide")
st.title("Campus Placement Assistant")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RAGCache:
    def __init__(self, cache_file="rag_cache.pkl"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.embedding_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
        
    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f)
        return {
            'embeddings': {},  # question -> embedding
            'responses': {},   # question -> response
            'metadata': {      # question -> metadata
                'timestamp': {},
                'hit_count': {}
            }
        }
    
    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
            
    def get_embedding(self, text):
        return self.embedding_model.get_text_embedding(text)
    
    def find_similar(self, query, threshold=0.95):
        query_embedding = self.get_embedding(query)
        
        for question, embedding in self.cache['embeddings'].items():
            similarity = self.compute_similarity(query_embedding, embedding)
            if similarity > threshold:
                self.cache['metadata']['hit_count'][question] += 1
                return self.cache['responses'][question]
        return None
    
    def compute_similarity(self, embed1, embed2):
        return sum(a * b for a, b in zip(embed1, embed2))
    
    def add_to_cache(self, question, response):
        self.cache['embeddings'][question] = self.get_embedding(question)
        self.cache['responses'][question] = response
        self.cache['metadata']['timestamp'][question] = time.time()
        self.cache['metadata']['hit_count'][question] = 1
        self.save_cache()
    
    def get_cache_stats(self):
        return {
            'size': len(self.cache['responses']),
            'total_hits': sum(self.cache['metadata']['hit_count'].values()),
            'most_common': sorted(
                self.cache['metadata']['hit_count'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        }
    
    def clear_cache(self):
        self.cache = {
            'embeddings': {},
            'responses': {},
            'metadata': {
                'timestamp': {},
                'hit_count': {}
            }
        }
        self.save_cache()

class CachedQueryEngine:
    def __init__(self, base_query_engine, cache):
        self.base_engine = base_query_engine
        self.cache = cache
        
    def query(self, query_str):
        # Try to get from cache first
        cached_response = self.cache.find_similar(query_str)
        if cached_response is not None:
            return cached_response
        
        # If not in cache, query the base engine
        response = self.base_engine.query(query_str)
        
        # Add to cache
        self.cache.add_to_cache(query_str, response)
        
        return response

def format_response_as_points(response_text):
    """Convert the response into bullet points"""
    sentences = [s.strip() for s in str(response_text).split('.') if s.strip()]
    formatted_response = "\n\n".join([f"• {sentence}." for sentence in sentences if sentence])
    return formatted_response

@st.cache_resource
def initialize_rag_system():
    """Initialize RAG system with document loading"""
    Settings.llm = Groq(
        model="llama3-70b-8192", 
        api_key=GROQ_API_KEY,
        context_window=8192,
        temperature=0.7,
        #system_prompt="Provide information based solely on the loaded documents. Structure responses as clear, concise points."
    )
    
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    #reader = SimpleDirectoryReader("./pl.txt", recursive=True)
    #documents = reader.load_data(show_progress=True)

    from llama_index.core import Document



    documents = SimpleDirectoryReader(
    input_files=["./campus-placement-procedure1.md"]
    ).load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    base_query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[
            LLMRerank(choice_batch_size=5, top_n=2),
        ],
        response_mode="tree_summarize",
    )
    
    rag_cache = RAGCache()
    cached_engine = CachedQueryEngine(base_query_engine, rag_cache)
    
    return cached_engine, rag_cache
    
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Initialize Qdrant client
    client = QdrantClient(path="./data")
    vector_store = QdrantVectorStore(client=client, collection_name="02_ReRanker_RAG")
    
    # Create index
    index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
    
    # Create base query engine
    base_query_engine = index.as_query_engine(
        similarity_top_k=10,
        node_postprocessors=[
            LLMRerank(choice_batch_size=5, top_n=2),
        ],
        response_mode="tree_summarize",
    )
    
    # Initialize cache
    rag_cache = RAGCache()
    
    # Create cached query engine
    cached_engine = CachedQueryEngine(base_query_engine, rag_cache)
    
    return cached_engine, rag_cache

# Initialize the RAG system
query_engine, rag_cache = initialize_rag_system()

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Type your queries here..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Get response from RAG system
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = query_engine.query(prompt)
            formatted_response = format_response_as_points(response)
            st.markdown(formatted_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": formatted_response})

# Sidebar with information and cache controls
with st.sidebar:
    st.title("About")
    st.markdown("""
    This is a RAG-powered chat assistant for campus placement procedures. 
    Ask questions about:
                
    - Placement process
    - Eligibility criteria
    - Interview procedures
    - Important dates
    - Required documents
    """)
    
    # Cache statistics
    st.divider()
    st.subheader("Cache Statistics")
    stats = rag_cache.get_cache_stats()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Cached Questions", stats['size'])
    with col2:
        st.metric("Total Cache Hits", stats['total_hits'])
    
    if stats['most_common']:
        st.subheader("Most Common Questions")
        for question, hits in stats['most_common']:
            st.text(f"({hits} hits) {question[:50]}...")
    
    # Cache controls
    st.divider()
    if st.button("Clear Cache"):
        rag_cache.clear_cache()
        st.success("Cache cleared!")
    
    # Chat history controls
    st.divider()
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()