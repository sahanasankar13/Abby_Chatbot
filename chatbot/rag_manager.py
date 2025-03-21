import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

class RAGManager:
    """Manages Retrieval Augmented Generation with efficient caching and retrieval"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the RAG manager
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.knowledge_base = []
        self.embeddings_cache = {}
        self.query_cache = {}
        self.cache_size = 1000  # Maximum number of cached embeddings
        
        # Required metadata fields for documents
        self.required_metadata = {'source_url', 'source_name', 'date_added'}
    
    def add_knowledge(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the knowledge base
        
        Args:
            documents (List[Dict[str, Any]]): List of documents with 'text' and 'metadata' keys
                metadata must include: source_url, source_name, date_added
        """
        # Validate documents have required metadata
        for doc in documents:
            if 'metadata' not in doc:
                raise ValueError("Document missing metadata field")
            missing_fields = self.required_metadata - set(doc['metadata'].keys())
            if missing_fields:
                raise ValueError(f"Document metadata missing required fields: {missing_fields}")
        
        self.knowledge_base.extend(documents)
        
        # Pre-compute embeddings for new documents
        for doc in documents:
            if doc['text'] not in self.embeddings_cache:
                self._add_to_cache(doc['text'])
    
    def query(self, 
              query: str, 
              top_k: int = 3,
              min_similarity: float = 0.6) -> List[Dict[str, Any]]:
        """
        Query the knowledge base
        
        Args:
            query (str): Query text
            top_k (int): Number of results to return
            min_similarity (float): Minimum similarity score (0-1) for results
            
        Returns:
            List[Dict[str, Any]]: Top k most relevant documents with metadata
        """
        # Check query cache first
        cache_key = f"{query}_{top_k}_{min_similarity}"
        if cache_key in self.query_cache:
            logger.debug("Query cache hit")
            return self.query_cache[cache_key]
        
        # Get query embedding
        query_embedding = self.model.encode([query])[0]
        
        # Get document embeddings
        doc_embeddings = []
        for doc in self.knowledge_base:
            if doc['text'] in self.embeddings_cache:
                doc_embeddings.append(self.embeddings_cache[doc['text']])
            else:
                embedding = self._add_to_cache(doc['text'])
                doc_embeddings.append(embedding)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Filter by minimum similarity and get top k
        valid_indices = np.where(similarities >= min_similarity)[0]
        top_indices = valid_indices[np.argsort(similarities[valid_indices])[-top_k:][::-1]]
        
        results = []
        for idx in top_indices:
            doc = self.knowledge_base[idx]
            results.append({
                'text': doc['text'],
                'metadata': doc['metadata'],
                'similarity': float(similarities[idx]),
                'citation': self._format_citation(doc['metadata'])
            })
        
        # Cache query results
        self.query_cache[cache_key] = results
        
        # Trim cache if needed
        if len(self.query_cache) > self.cache_size:
            oldest_keys = list(self.query_cache.keys())[:-self.cache_size]
            for key in oldest_keys:
                del self.query_cache[key]
        
        return results
    
    def _format_citation(self, metadata: Dict[str, Any]) -> str:
        """Format metadata into a citation string"""
        return f"Source: {metadata['source_name']} ({metadata['source_url']})"
    
    def get_source_url(self, doc_text: str) -> Optional[str]:
        """Get source URL for a document if it exists in knowledge base"""
        for doc in self.knowledge_base:
            if doc['text'] == doc_text:
                return doc['metadata'].get('source_url')
        return None
    
    def _add_to_cache(self, text: str) -> np.ndarray:
        """
        Add text embedding to cache
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        embedding = self.model.encode([text])[0]
        self.embeddings_cache[text] = embedding
        
        # Trim cache if needed
        if len(self.embeddings_cache) > self.cache_size:
            # Remove oldest entries
            oldest_keys = list(self.embeddings_cache.keys())[:-self.cache_size]
            for key in oldest_keys:
                del self.embeddings_cache[key]
        
        return embedding
    
    def clear_caches(self) -> None:
        """Clear all caches"""
        self.embeddings_cache.clear()
        self.query_cache.clear()
        logger.info("RAG caches cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """
        Get cache statistics
        
        Returns:
            Dict[str, int]: Cache statistics
        """
        return {
            'embeddings_cache_size': len(self.embeddings_cache),
            'query_cache_size': len(self.query_cache),
            'knowledge_base_size': len(self.knowledge_base)
        } 