import os
import logging
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import faiss
from utils.data_loader import load_reproductive_health_data

logger = logging.getLogger(__name__)

class BertRAGModel:
    """
    BERT-based Retrieval-Augmented Generation model for reproductive health information
    """
    def __init__(self):
        """Initialize the BERT RAG model with pre-trained embeddings"""
        logger.info("Initializing BERT RAG Model")
        try:
            # Load pre-trained model and tokenizer
            self.model_name = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            
            # Load and index the data
            self.qa_pairs = load_reproductive_health_data()
            self.build_index()
            
            logger.info("BERT RAG Model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing BERT RAG Model: {str(e)}", exc_info=True)
            raise
    
    def build_index(self):
        """Build FAISS index from question-answer pairs"""
        logger.info("Building FAISS index for RAG model")
        try:
            # Extract questions and generate embeddings
            questions = [qa['Question'] for qa in self.qa_pairs]
            self.question_embeddings = self.generate_embeddings(questions)
            
            # Create FAISS index
            self.dimension = self.question_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(self.question_embeddings)
            
            logger.info(f"FAISS index built with {len(questions)} questions")
        except Exception as e:
            logger.error(f"Error building FAISS index: {str(e)}", exc_info=True)
            raise
    
    def generate_embeddings(self, texts):
        """
        Generate embeddings for a list of texts using BERT
        
        Args:
            texts (list): List of text strings to embed
        
        Returns:
            numpy.ndarray: Normalized embeddings
        """
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and get BERT embeddings
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use mean pooling to get sentence embeddings
            attention_mask = inputs['attention_mask']
            mean_embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
            
            # Normalize
            normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
            embeddings.append(normalized_embeddings.numpy())
        
        return np.vstack(embeddings)
    
    def _mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on token embeddings
        
        Args:
            token_embeddings: Token embeddings from BERT
            attention_mask: Attention mask for padding
        
        Returns:
            torch.Tensor: Mean-pooled embeddings
        """
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    def get_response(self, question, top_k=3):
        """
        Get response for a given question using RAG
        
        Args:
            question (str): The question to answer
            top_k (int): Number of similar questions to retrieve
        
        Returns:
            str: The answer to the question
        """
        try:
            # Generate embedding for the question
            question_embedding = self.generate_embeddings([question])
            
            # Search for similar questions
            distances, indices = self.index.search(question_embedding, top_k)
            
            # Get the most similar question's answer
            best_idx = indices[0][0]
            best_answer = self.qa_pairs[best_idx]['Answer']
            
            logger.debug(f"RAG found answer with distance: {distances[0][0]}")
            return best_answer
        
        except Exception as e:
            logger.error(f"Error getting RAG response: {str(e)}", exc_info=True)
            return "I'm sorry, I couldn't find a good answer to your question."
    
    def is_confident(self, question, response, threshold=10.0):
        """
        Determine if the RAG model is confident in its response
        
        Args:
            question (str): The original question
            response (str): The generated response
            threshold (float): Confidence threshold (lower is more confident)
        
        Returns:
            bool: True if confident, False otherwise
        """
        try:
            # Generate embedding for the question
            question_embedding = self.generate_embeddings([question])
            
            # Search for the closest question
            distances, _ = self.index.search(question_embedding, 1)
            
            # Lower distance means higher confidence
            confidence = distances[0][0]
            logger.debug(f"Confidence score: {confidence}")
            
            return confidence < threshold
        
        except Exception as e:
            logger.error(f"Error checking confidence: {str(e)}", exc_info=True)
            return False
