import nest_asyncio
import logging
import os                                
import glob
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
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import Document
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_index.core.memory import ChatMemoryBuffer
import enum
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Add these imports right after your existing imports (around line 28)
from llama_index.core.evaluation import (
    CorrectnessEvaluator,
    RelevancyEvaluator,
    FaithfulnessEvaluator,
    SemanticSimilarityEvaluator,
    PairwiseComparisonEvaluator,
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    BatchEvalRunner
)
from llama_index.core.evaluation.eval_utils import get_responses
import pandas as pd
from typing import List, Dict, Any
import json
import copy

initial_phase = "near_placement"
current_phase = initial_phase

class PlacementPhaseHandler:
    def __init__(self):
        self.placement_phases = {
            "early_preparation": {
                "keywords": [
                    "skill development", "resume building", "portfolio creation", 
                    "learning programming", "internship preparation", 
                    "academic performance", "extracurricular activities"
                ],
                "system_prompt": """
                You are a campus placement mentor focusing on early-stage preparation. 
                Provide strategic guidance for students who have plenty of time before placements. 
                Focus on:
                1. Long-term skill development
                2. Building a strong academic and project portfolio
                3. Understanding industry trends
                4. Exploring internship opportunities
                5. Holistic personal and professional growth strategies
                Always provide actionable, forward-looking advice that helps students 
                systematically prepare for future placement opportunities.
                """,
                "retrieval_config": {
                    "similarity_top_k": 15,  # Broader, more comprehensive retrieval
                    "diversity_weight": 0.3  # Encourage diverse content
                }
            },
            "near_placement": {
                "keywords": [
                    "interview preparation", "company research", "mock interviews", 
                    "technical interview", "soft skills", "placement strategy", 
                    "resume finalization", "aptitude preparation"
                ],
                "system_prompt": """
                You are a placement preparation coach for students approaching their placement season. 
                Provide focused, tactical advice on:
                1. Interview techniques and strategies
                2. Technical and soft skill refinement
                3. Company-specific preparation
                4. Resume optimization
                5. Immediate skill enhancement tactics
                Deliver precise, actionable guidance that helps students 
                maximize their placement readiness in the short term.
                """,
                "retrieval_config": {
                    "similarity_top_k": 10,  # More focused retrieval
                    "diversity_weight": 0.2  # Balance between precision and variety
                }
            },
            "post_placement": {
                "keywords": [
                    "onboarding", "first job", "professional growth", 
                    "career development", "workplace adaptation", 
                    "professional networking", "skill progression"
                ],
                "system_prompt": """
                You are a career transition mentor for students who have secured their placement. 
                Provide comprehensive guidance on:
                1. Smooth workplace transition
                2. Initial job performance strategies
                3. Professional networking
                4. Continued learning and skill development
                5. Long-term career planning
                Offer supportive, forward-looking advice that helps new professionals 
                navigate their initial career phase effectively.
                """,
                "retrieval_config": {
                    "similarity_top_k": 8,  # More curated, specific retrieval
                    "diversity_weight": 0.1  # Highly targeted content
                }
            }
        }
    
    def get_phase_configuration(self, phase):
        """
        Retrieve the configuration for a specific placement phase.
        
        Args:
            phase (str): The placement phase to retrieve
        
        Returns:
            dict: Configuration for the specified phase
        """
        return self.placement_phases.get(phase, self.placement_phases["near_placement"])
    
    def augment_query(self, query, phase):
        """
        Augment the query with phase-specific context and keywords.
        
        Args:
            query (str): Original user query
            phase (str): Current placement phase
        
        Returns:
            str: Augmented query with phase-specific context
        """
        phase_config = self.get_phase_configuration(phase)
        
        # Add phase-specific context to the query
        augmented_query = (
            f"Placement Phase Context: {phase.replace('_', ' ').title()} Phase\n"
            f"Phase Keywords: {', '.join(phase_config['keywords'])}\n\n"
            f"Original Query: {query}"
        )
        
        return augmented_query

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class RAGCache:
    def __init__(self, cache_file="rag_cache.pkl"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
        self.embedding_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    def _load_cache(self):
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                
                # Ensure all required keys exist in loaded cache
                if 'embeddings' not in cache:
                    cache['embeddings'] = {}
                if 'responses' not in cache:
                    cache['responses'] = {}
                if 'metadata' not in cache:
                    cache['metadata'] = {'timestamp': {}, 'hit_count': {}}
                elif 'timestamp' not in cache['metadata']:
                    cache['metadata']['timestamp'] = {}
                elif 'hit_count' not in cache['metadata']:
                    cache['metadata']['hit_count'] = {}
                if 'document_mapping' not in cache:
                    cache['document_mapping'] = {}
                if 'document_paths' not in cache:
                    cache['document_paths'] = {}
                
                return cache
            except Exception as e:
                print(f"Error loading cache: {e}")
                # Return fresh cache if loading fails
                return self._create_empty_cache()
        return self._create_empty_cache()
    
    def _create_empty_cache(self):
        """Create a new empty cache with all required keys."""
        return {
            'embeddings': {},  # question -> embedding
            'responses': {},   # question -> response
            'metadata': {      # question -> metadata
                'timestamp': {},
                'hit_count': {}
            },
            'document_mapping': {},  # document_id -> [list of queries]
            'document_paths': {}     # document_id -> file_path
        }

    def save_cache(self):
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
            
    def get_embedding(self, text):
        return self.embedding_model.get_text_embedding(text)
    
    def find_similar(self, query, recent_history=None, threshold=0.92):
        # Create a context-aware query by including recent history
        if recent_history and len(recent_history) > 0:
            # Use the last message as context
            context_query = f"Previous: {recent_history[-1]}. Current: {query}"
        else:
            context_query = query
            
        query_embedding = self.get_embedding(context_query)
        
        best_match = None
        best_similarity = -1
        
        for question, embedding in self.cache['embeddings'].items():
            similarity = self.compute_cosine_similarity(query_embedding, embedding)
            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = question
        
        if best_match:
            self.cache['metadata']['hit_count'][best_match] += 1
            self.save_cache()
            return self.cache['responses'][best_match]
        
        return None
    
    def compute_cosine_similarity(self, embed1, embed2):
        """Compute cosine similarity between two embedding vectors."""
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embed1, embed2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embed1) ** 0.5
        magnitude2 = sum(b * b for b in embed2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
            
        # Return cosine similarity
        return dot_product / (magnitude1 * magnitude2)
    
    def add_to_cache(self, question, response, recent_history=None, document_id="unknown_doc"):
        """Add query to cache with recent history for context."""
        # Create a context-aware query
        if recent_history and len(recent_history) > 0:
            context_question = f"Previous: {recent_history[-1]}. Current: {question}"
        else:
            context_question = question
            
        self.cache['embeddings'][context_question] = self.get_embedding(context_question)
        self.cache['responses'][context_question] = response
        self.cache['metadata']['timestamp'][context_question] = time.time()
        self.cache['metadata']['hit_count'][context_question] = 1
        
        # Track document associations
        if document_id not in self.cache['document_mapping']:
            self.cache['document_mapping'][document_id] = []
        self.cache['document_mapping'][document_id].append(context_question)
        
        self.save_cache()

    def register_document(self, doc_id, doc_path):
        """Register a document in the cache with its file path"""
        if 'document_paths' not in self.cache:
            self.cache['document_paths'] = {}
        
        self.cache['document_paths'][doc_id] = doc_path
        self.save_cache()

    def get_registered_documents(self):
        """Get dictionary of registered document IDs to file paths"""
        if 'document_paths' not in self.cache:
            self.cache['document_paths'] = {}
        
        return self.cache['document_paths']

    def remove_queries_for_document(self, document_id):
        """Remove queries linked to a specific document when it's updated or deleted."""
        if 'document_mapping' not in self.cache:
            self.cache['document_mapping'] = {}
            return 0
            
        if document_id in self.cache['document_mapping']:
            queries_to_remove = self.cache['document_mapping'][document_id]
            for query in queries_to_remove:
                self.cache['embeddings'].pop(query, None)
                self.cache['responses'].pop(query, None)
                self.cache['metadata']['timestamp'].pop(query, None)
                self.cache['metadata']['hit_count'].pop(query, None)
            
            # Remove document entry
            del self.cache['document_mapping'][document_id]
            
            # Remove from document paths if it exists
            if 'document_paths' in self.cache and document_id in self.cache['document_paths']:
                del self.cache['document_paths'][document_id]
            
            self.save_cache()
            return len(queries_to_remove)
        return 0

    def get_cache_stats(self):
        """Get statistics about the cache with safe access to all keys."""
        # Ensure all required keys exist
        if 'document_mapping' not in self.cache:
            self.cache['document_mapping'] = {}
        if 'document_paths' not in self.cache:
            self.cache['document_paths'] = {}
        if 'metadata' not in self.cache:
            self.cache['metadata'] = {'timestamp': {}, 'hit_count': {}}
        if 'hit_count' not in self.cache['metadata']:
            self.cache['metadata']['hit_count'] = {}
        
        return {
            'size': len(self.cache['responses']),
            'total_hits': sum(self.cache['metadata']['hit_count'].values()),
            'most_common': sorted(
                self.cache['metadata']['hit_count'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            'documents': {
                path: len(self.cache['document_mapping'].get(doc_id, []))
                for doc_id, path in self.get_registered_documents().items()
            }
        }
    
    def clear_cache(self):
        self.cache = self._create_empty_cache()
        self.save_cache()

# CachedQueryEngine remains unchanged
class CachedQueryEngine:
    def __init__(self, base_query_engine, cache):
        self.base_engine = base_query_engine
        self.cache = cache

    def query(self, query_str, document_id="unknown_doc"):
        # Try to get from cache first
        cached_response = self.cache.find_similar(query_str)
        if cached_response is not None:
            return cached_response

        # If not in cache, query the base engine
        response = self.base_engine.query(query_str)
        
        # Try to identify document source
        if hasattr(response, 'source_nodes') and response.source_nodes:
            # Extract document IDs from source nodes
            doc_ids = []
            for node in response.source_nodes:
                if hasattr(node, 'node') and hasattr(node.node, 'metadata'):
                    file_path = node.node.metadata.get('file_path', None)
                    if file_path:
                        # Create a stable doc_id based on file path
                        doc_id = str(hash(file_path))
                        doc_ids.append(doc_id)
                        # Register this document
                        self.cache.register_document(doc_id, file_path)
            
            if doc_ids:
                # Use the first source document as the primary source
                document_id = doc_ids[0]
        
        # Add to cache with document ID
        self.cache.add_to_cache(query_str, response, document_id)

        return response
    
class FollowUpQuestionAgent:
    def __init__(self):
        self.follow_up_indicators = [
            "but how", "how exactly", "tell me more", "elaborate on", "explain further",
            "give me details", "can you explain", "what about", "how do i", "how would i",
            "but what if", "but help me", "but show me", "how specifically", "why is that",
            "could you clarify", "i don't understand", "what does that mean", "in what way",
            "when you say", "you mentioned", "referring to", "regarding that", "on that note",
            "additionally", "furthermore", "also", "so", "and", "then", "after that",
            "besides", "moreover", "in relation to", "related to", "in that case"
        ]
        
        self.pronoun_references = [
            "it", "this", "that", "these", "those", "they", "them", "he", "she", 
            "his", "her", "their", "its", "there"
        ]
        
def is_follow_up(self, current_query, chat_history):
    """
    Determines if the current query is a follow-up question based on chat history.
    
    Args:
        current_query (str): The current user query
        chat_history (list): List of previous messages
        
    Returns:
        bool: True if it's a follow-up question, False otherwise
        str: The augmented query if it's a follow-up, otherwise the original query
    """
    # Embedding model for cosine similarity calculation
    from llama_index.embeddings.fastembed import FastEmbedEmbedding
    
    # Initialize embedding model
    embedding_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    def compute_cosine_similarity(embed1, embed2):
        """Compute cosine similarity between two embedding vectors."""
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(embed1, embed2))
        
        # Calculate magnitudes
        magnitude1 = sum(a * a for a in embed1) ** 0.5
        magnitude2 = sum(b * b for b in embed2) ** 0.5
        
        # Avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
            
        # Return cosine similarity
        return dot_product / (magnitude1 * magnitude2)
    
    if not chat_history or len(chat_history) < 2:
        return False, current_query
        
    # Get the most recent query and response
    prev_messages = [msg for msg in chat_history[-4:] if isinstance(msg, dict)]
    
    # Extract just the last user query and assistant response
    last_user_query = None
    last_assistant_response = None
    
    # Find the last user query and assistant response
    for msg in reversed(prev_messages):
        if msg.get("role") == "user" and not last_user_query:
            last_user_query = msg.get("content", "")
        elif msg.get("role") == "assistant" and not last_assistant_response:
            last_assistant_response = msg.get("content", "")
        
        if last_user_query and last_assistant_response:
            break
            
    if not last_user_query or not last_assistant_response:
        return False, current_query
        
    # Normalize queries for comparison
    current_query_lower = current_query.lower().strip()
    last_query_lower = last_user_query.lower().strip()
    
    # Compute embeddings for analysis
    current_query_embed = embedding_model.get_text_embedding(current_query_lower)
    last_query_embed = embedding_model.get_text_embedding(last_query_lower)
    
    # Compute cosine similarity between current and last query
    query_similarity = compute_cosine_similarity(current_query_embed, last_query_embed)
    
    # Compute embeddings for follow-up indicators
    follow_up_embeds = [embedding_model.get_text_embedding(indicator.lower()) for indicator in self.follow_up_indicators]
    current_query_embed = embedding_model.get_text_embedding(current_query_lower)
    
    # Check similarity with follow-up indicators
    indicator_similarities = [compute_cosine_similarity(current_query_embed, indicator_embed) for indicator_embed in follow_up_embeds]
    max_indicator_similarity = max(indicator_similarities) if indicator_similarities else 0
    
    # Enhanced follow-up detection criteria
    is_explicit_follow_up = max_indicator_similarity > 0.7
    is_context_similar = query_similarity > 0.6
    
    # Check for pronoun references which suggest this is continuing from prior context
    has_pronoun_reference = any(
        f" {pronoun} " in f" {current_query_lower} " for pronoun in self.pronoun_references
    )
    
    # Check for very short queries that likely rely on context
    is_short_query = len(current_query_lower.split()) <= 5
    
    # Check for queries without context (e.g., "Why?" or "How?")
    is_contextless_query = (
        len(current_query_lower.split()) <= 2 and 
        any(q in current_query_lower for q in ["why", "how", "what", "when", "where", "who"])
    )
    
    # Combine detection criteria
    is_follow_up = (
        is_explicit_follow_up or 
        is_context_similar or 
        has_pronoun_reference or 
        is_contextless_query
    )
    
    # If it's a follow-up, augment the query with context
    if is_follow_up:
        # Create contextually enhanced query
        augmented_query = f"Previous question: '{last_user_query}'. Previous answer: '{last_assistant_response}'. Current question: '{current_query}'"
        return True, augmented_query
        
    return False, current_query    
        
    def augment_query_with_history(self, query, chat_history, max_history=2):
        """
        Create a more comprehensive context by including recent chat history.
        This helps the RAG system understand the broader conversation context.
        """
        if not chat_history or len(chat_history) < 2:
            return query
            
        recent_exchanges = []
        history_count = 0
        
        for msg in reversed(chat_history):
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                role = msg["role"]
                content = msg["content"]
                recent_exchanges.append(f"{role.capitalize()}: {content}")
                history_count += 0.5  # Count each message as half an exchange
                
                if history_count >= max_history:
                    break
        
        if not recent_exchanges:
            return query
            
        # Reverse to get chronological order
        recent_exchanges.reverse()
        
        # Create context-enhanced query
        context = " ".join(recent_exchanges)
        augmented_query = f"Chat history context: {context}\n\nCurrent question: {query}"
        
        return augmented_query

class CachedChatEngine:
    def __init__(self, base_chat_engine, cache):
        self.base_engine = base_chat_engine
        self.cache = cache
    
        self.history = []  # Keep track of recent messages
        self.follow_up_agent = FollowUpQuestionAgent()  # Ensure this line is executed
        
        # List of common greetings and simple queries that shouldn't be cached
        self.non_cacheable_phrases = [
            "hi", "hello", "hey", "hi there", "hello there", "greetings", 
            "good morning", "good afternoon", "good evening", "howdy",
            "what's up", "how are you", "how's it going", "nice to meet you",
            "thanks", "thank you", "ok", "okay", "sure", "yes", "no",
            "bye", "goodbye", "see you", "talk to you later"
        ]

    def is_cacheable_query(self, message):
        """Determine if a message should be cached based on content analysis."""
        # Convert to lowercase for comparison
        message_lower = message.lower().strip()
        
        # 2. Check for follow-up questions - these should never be cached
        follow_up_indicators = [
            "but how", "how exactly", "tell me more", "elaborate on", "explain further",
            "give me details", "can you explain", "what about", "how do i", "how would i",
            "but what if", "but help me", "but show me", "how specifically"
        ]
        
        if (len(self.history) > 0 and 
            any(indicator in message_lower for indicator in follow_up_indicators)):
            return False
        
        # 3. Check if it's a substantive question about a specific topic that should be cached
        # Even if it uses the word "summarize" but is substantive
        substantive_topic_indicators = [
            "how to", "steps to", "process for", "method for", "approach to", 
            "ways to", "techniques for", "strategies for", "best practices", 
            "improve my", "develop my", "enhance my", "skill set", "skills"
        ]
        
        # If it's a question about a substantive topic, it should be cached
        # regardless of whether it contains the word "summarize"
        if any(indicator in message_lower for indicator in substantive_topic_indicators):
            return True
        
        # 4. Very short queries are not cacheable
        if len(message_lower.split()) < 5:
            return False
        
        # 5. Check for questions asking Claude to ask something
        if ("ask me" in message_lower or "can you ask" in message_lower) and "question" in message_lower:
            return False
        
        # 6. Check for simple "can you X?" questions
        if message_lower.startswith("can you") and len(message_lower.split()) < 6:
            return False
        
        # 7. Check word diversity/information content
        unique_words = set(message_lower.split())
        word_diversity_ratio = len(unique_words) / len(message_lower.split()) if message_lower.split() else 0
        if word_diversity_ratio < 0.5 and len(message_lower.split()) < 8:
            return False
        
        # 8. Simple questions with just question marks
        if message_lower.count("?") > 0 and len(message_lower) < 15:
            return False
                
        # 9. Check for information-seeking content
        information_seeking_words = ["how", "what", "why", "when", "where", "which", "who", "explain", 
                                    "describe", "tell", "detail", "elaborate", "define"]
        has_information_seeking_content = any(word in message_lower.split() for word in information_seeking_words)
        
        # Short queries without information seeking words are likely not cacheable
        if not has_information_seeking_content and len(message_lower.split()) < 7:
            return False
        
        # Consider it a substantive query worth caching
        return True

    def chat(self, message, document_id="unknown_doc"):
        # Get chat history for context
        chat_history = self.history
        
        # Check if this is a follow-up question using the agent
        is_follow_up, processed_message = self.follow_up_agent.is_follow_up(message, chat_history)
        
        # Always update history first for proper context
        self.history.append(message)
        if len(self.history) > 5:  # Keep last 5 messages for context
            self.history.pop(0)
        
        # Detect if the message is a follow-up or should be cached
        is_cacheable = self.is_cacheable_query(message) and not is_follow_up
        
        # For cacheable queries, try cache first
        if is_cacheable:
            cached_response = self.cache.find_similar(message, self.history[-2:] if len(self.history) > 1 else None)
            if cached_response is not None:
                return cached_response
        
        # Use the original message for cache lookups but processed message for querying
        query_message = processed_message if is_follow_up else message
        
        # If not in cache or not cacheable, query the base engine
        response = self.base_engine.chat(query_message)
        
        # Only cache substantive queries that are determined to be cacheable
        # Don't cache the augmented query, but the original one
        if is_cacheable:
            self.cache.add_to_cache(message, response, self.history[-2:] if len(self.history) > 1 else None, document_id)
        
        return response
        
    def reset(self):
        # Reset the chat history
        self.history = []
        if hasattr(self.base_engine, "reset"):
            self.base_engine.reset()

def format_response_as_points(response):
    """Convert the response into well-structured bullet points"""
    # Convert response to string
    response_text = str(response)
    
    # If the response has a 'response' attribute (common in some RAG systems)
    if hasattr(response, 'response'):
        response_text = str(response.response)
    
    # Remove any existing bullet points or numbering
    response_text = response_text.replace('•', '').replace('-', '').strip()
    
    # Split into sentences, but be smarter about it
    sentences = []
    current_sentence = ""
    for part in response_text.split('.'):
        part = part.strip()
        if not part:
            continue
        
        # Check if the part is too short to be a complete point
        if len(part.split()) < 3:
            current_sentence += f". {part}"
        else:
            if current_sentence:
                sentences.append(current_sentence)
                current_sentence = part
            else:
                current_sentence = part
    
    # Add the last sentence
    if current_sentence:
        sentences.append(current_sentence)
    
    # Format as bullet points, combining short points if necessary
    formatted_points = []
    current_point = ""
    for sentence in sentences:
        if not current_point or len(current_point.split()) < 10:
            current_point += f" {sentence.strip()}."
        else:
            formatted_points.append(f"• {current_point.strip()}")
            current_point = sentence.strip() + "."
    
    # Add the last point
    if current_point:
        formatted_points.append(f"• {current_point.strip()}")
    
    return "\n\n".join(formatted_points)

def update_query_engine(base_query_engine, phase_handler, phase):
    """
    Dynamically update the query engine based on placement phase.
    
    Args:
        base_query_engine: Original query engine
        phase_handler (PlacementPhaseHandler): Phase configuration handler
        phase (str): Current placement phase
    
    Returns:
        Modified query engine with phase-specific configuration
    """
    # Get phase-specific configuration
    phase_config = phase_handler.get_phase_configuration(phase)

    # Instead of deep copying, create a new instance with the same configuration
    # This avoids trying to pickle the ONNX runtime session
    try:
        # Create a new query engine with similar configuration
        modified_engine = base_query_engine.__class__(
            retriever=base_query_engine.retriever,
            response_synthesizer=base_query_engine._response_synthesizer
        )
        
        # Copy over other important attributes
        if hasattr(base_query_engine, 'node_postprocessors'):
            modified_engine.node_postprocessors = base_query_engine.node_postprocessors.copy()
        if hasattr(base_query_engine, 'callback_manager'):
            modified_engine.callback_manager = base_query_engine.callback_manager
    except Exception as e:
        print(f"Could not create new query engine instance: {str(e)}")
        # Fall back to using the original engine if creation fails
        modified_engine = base_query_engine
    
    # Update retriever settings
    if hasattr(modified_engine, 'retriever'):
        modified_engine.retriever.similarity_top_k = phase_config['retrieval_config'].get('similarity_top_k', 10)
    
    # Update system prompt if possible
    try:
        if hasattr(modified_engine, "update_prompts"):
            modified_engine.update_prompts({
                "system_prompt": phase_config['system_prompt']
            })
    except Exception as e:
        print(f"Could not update system prompt: {str(e)}")
    
    return modified_engine

def initialize_rag_system():
    """Initialize RAG system with document loading strictly from data folder"""
    # Initialize RAG Cache first
    rag_cache = RAGCache()
    
    # Configure LLM and embedding settings
    Settings.llm = Groq(
        model="llama3-70b-8192", 
        api_key=GROQ_API_KEY,
        context_window=8192,
        temperature=0.7,
    )
    
    Settings.embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # STRICT DOCUMENT LOADING - Only from data folder
    try:
        # Define the exact path to the data folder
        data_dir = "./data"
        
        # Verify the data directory exists
        if not os.path.exists(data_dir) or not os.path.isdir(data_dir):
            print(f"Data directory not found: {data_dir}")
            documents = [Document(text="No documents available", id_="empty_doc")]
        else:
            # Find all files in the data directory only (no parent directory files)
            valid_extensions = [".pdf", ".txt", ".docx", ".md", ".pptx"]
            file_list = []
            
            # Use glob to get only files in the data directory with valid extensions
            for ext in valid_extensions:
                # Make sure we're only getting files from the data directory
                pattern = os.path.join(data_dir, f"*{ext}")
                found_files = glob.glob(pattern)
                file_list.extend(found_files)
                
                # Include files in subdirectories if needed
                pattern = os.path.join(data_dir, f"/*{ext}")
                found_files = glob.glob(pattern, recursive=True)
                file_list.extend(found_files)
            
            # Remove any duplicates
            file_list = list(set(file_list))
            
            # Load documents manually to ensure we only get data folder files
            documents = []
            for file_path in file_list:
                try:
                    # Use SimpleDirectoryReader for individual files
                    doc = SimpleDirectoryReader(input_files=[file_path]).load_data()
                    
                    # If loaded successfully, add to our document list
                    if doc:
                        for d in doc:
                            # Make sure metadata contains the file path
                            if not hasattr(d, 'metadata'):
                                d.metadata = {}
                            
                            d.metadata['file_path'] = file_path
                            documents.append(d)
                except Exception as e:
                    print(f"Error loading {os.path.basename(file_path)}: {str(e)}")
            
            # Create document ID based on file path
            for i, doc in enumerate(documents):
                if not hasattr(doc, 'id_') or not doc.id_:
                    file_path = doc.metadata.get('file_path', f"unknown_{i}")
                    doc.id_ = f"doc_{hash(file_path)}_{i}"
            
            # Success message with actual count of files (not chunks)
            print(f"Loaded {len(file_list)} documents from data folder")
    
    except Exception as e:
        print(f"Error loading documents: {str(e)}")
        documents = [Document(text="No documents available", id_="empty_doc")]
    
    # Convert to nodes (this is where chunks are created)
    node_parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    
    # Process documents to create nodes
    nodes = []
    for doc in documents:
        # Extract the text from the document
        text = doc.text if hasattr(doc, 'text') else str(doc)
        
        # Get document metadata
        metadata = {}
        if hasattr(doc, 'metadata') and isinstance(doc.metadata, dict):
            metadata = doc.metadata.copy()
        
        # Parse the text into chunks
        doc_nodes = node_parser.get_nodes_from_documents([Document(text=text, metadata=metadata, id_=getattr(doc, 'id_', f"doc_{len(nodes)}"))])
        nodes.extend(doc_nodes)
    
    # Create index from our manually processed nodes
    index = VectorStoreIndex(nodes)
    
    memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
    
    # Create a chat engine instead of a query engine
    
    # Add a count of how many chunks were created
    print(f"Created {len(nodes)} text chunks for processing")
    
    # Create a standard query engine
    standard_query_engine = index.as_query_engine()

    phase_handler = PlacementPhaseHandler()
    
    # Create a retriever with better context capabilities
    vector_retriever = index.as_retriever(
        similarity_top_k=12,
    )
    
    # Create a response synthesizer
    _response_synthesizer = get_response_synthesizer(
        response_mode="compact",
        verbose=True
    )
    
    # Create the query engine with the explicitly created synthesizer

    base_query_engine = update_query_engine(standard_query_engine, phase_handler, initial_phase)
    phase_config = phase_handler.get_phase_configuration(initial_phase)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        memory=memory,
        system_prompt=phase_config['system_prompt'],
        similarity_top_k=phase_config['retrieval_config'].get('similarity_top_k', 10)
    )
    # Add a system prompt
    try:
        if hasattr(base_query_engine, "update_prompts"):
            base_query_engine.update_prompts({
                "system_prompt": """
                You are a helpful campus placement assistant. When answering questions, consider context from 
                multiple slides or pages. 
                """
            })
    except Exception as e:
        print(f"Could not update system prompt: {str(e)}")
    
    cached_engine = CachedChatEngine(chat_engine, rag_cache)
    return cached_engine, rag_cache,phase_handler




# Initialize the RAG system
# Initialize the system
chat_engine, rag_cache, rag_evaluator, phase_handler = initialize_rag_system()

# Store messages in a regular list instead of session state
messages = []

def display_messages():
    """Print all messages in the conversation history"""
    for message in messages:
        print(f"[{message['role']}]: {message['content']}")

def handle_user_input(prompt):
    """Process user input and generate a response"""
    # Print user message
    print(f"[user]: {prompt}")
    
    # Add user message to chat history
    messages.append({"role": "user", "content": prompt})
    
    # Check if it's a follow-up question
    follow_up_agent = FollowUpQuestionAgent()
    is_follow_up, processed_prompt = follow_up_agent.is_follow_up(
        prompt, 
        messages[:-1]  # Exclude the message we just added
    )
    
    # Get response from RAG system
    print("Thinking...")
    augmented_prompt = phase_handler.augment_query(prompt, current_phase)
    if hasattr(chat_engine, 'base_engine'):
        chat_engine.base_engine = update_query_engine(
            chat_engine.base_engine, 
            phase_handler, 
            current_phase
        )
    
    # Use the processed prompt that includes context if it's a follow-up
    response = chat_engine.chat(prompt)  # The chat method now handles follow-up detection
    formatted_response = format_response_as_points(response)
    
    # Optionally show a small indicator that this was a follow-up question
    if is_follow_up:
        print("Follow-up question detected - Using conversation context")
        
    print(f"[assistant]: {formatted_response}")
    
    # Add assistant response to chat history
    messages.append({"role": "assistant", "content": formatted_response})
    
    return formatted_response



def display_cache_stats():
    """Display cache statistics"""
    print("\n==== Cache Statistics ====")
    stats = rag_cache.get_cache_stats()
    
    print(f"Cached Questions: {stats['size']}")
    print(f"Total Cache Hits: {stats['total_hits']}")
    
    if stats['most_common']:
        print("\nMost Common Questions:")
        for question, hits in stats['most_common']:
            print(f"({hits} hits) {question[:50]}...")
    
    # Document statistics - SHOW ONLY DATA FOLDER FILES
    if stats.get('documents'):
        print("\nDocument Sources:")
        for doc_path, count in stats['documents'].items():
            # Only show files from data folder by checking path
            if os.path.dirname(doc_path) == "data" or os.path.dirname(doc_path) == "./data" or doc_path.startswith("data/") or doc_path.startswith("./data/"):
                # Get just the filename from the path
                filename = os.path.basename(doc_path) if doc_path != "unknown_file" else "Unknown Source"
                print(f"{filename}: {count} queries")

def clear_cache():
    """Clear the RAG cache"""
    rag_cache.clear_cache()
    print("Cache cleared!")

def clear_chat_history():
    """Clear the chat history"""
    messages.clear()
    print("Chat history cleared!")

def reset_chat():
    """Reset the chat engine and clear history"""
    chat_engine.reset()
    messages.clear()
    print("Chat reset!")

# Example of how to use the system
def main_interaction():
    # Initialize the system with a default phase
    chat_engine, rag_cache, rag_evaluator, phase_handler = initialize_rag_system()
    
    # Current placement phase (can be changed by user)
    current_phase = "near_placement"
    
    while True:
        user_input = input(
            "\nSelect an option:\n"
            "1. Change Placement Phase\n"
            "2. Ask a Question\n"
            "3. View Cache Stats\n"
            "4. Clear Cache\n"
            "5. Reset Chat\n"
            "6. Run RAG Evaluation\n"
            "7. Exit\n"
            "Enter your choice: "
        )
        
        if user_input == "1":
            # Phase selection
            print("\nSelect Placement Phase:")
            print("1. Early Preparation (Lot of time left)")
            print("2. Near Placement")
            print("3. Post Placement")
            
            phase_choice = input("Enter phase number: ")
            phase_map = {
                "1": "early_preparation",
                "2": "near_placement", 
                "3": "post_placement"
            }
            current_phase = phase_map.get(phase_choice, "near_placement")
            
            # Reinitialize the system with the new phase
            chat_engine, rag_cache, rag_evaluator, phase_handler = initialize_rag_system(current_phase)
            print(f"Switched to {current_phase.replace('_', ' ').title()} phase.")
        
        elif user_input == "2":
            query = input("Enter your query: ")
            response = handle_user_input(query, current_phase, chat_engine, phase_handler)
            print(response)
        
        elif user_input == "3":
            display_cache_stats()
        
        elif user_input == "4":
            clear_cache()
        
        elif user_input == "5":
            reset_chat()
        
        elif user_input == "7":
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main_interaction()
