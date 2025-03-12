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

    def _is_conversational_query(self, question):
        """Check if the query is conversational rather than informational."""
        question = question.lower()
        greetings = ["hi", "hello", "hey", "greetings", "howdy", "how are you"]
        goodbyes = ["bye", "goodbye", "see you", "farewell", "exit", "quit"]

        # Check if any greeting or goodbye is in the question
        is_greeting = any(greeting in question for greeting in greetings)
        is_goodbye = any(goodbye in question for goodbye in goodbyes)

        if is_goodbye:
            return "goodbye"
        elif is_greeting:
            return "greeting"
        return False

    def get_response(self, question, top_k=5):
        """
        Get response for a given question using RAG

        Args:
            question (str): The question to answer
            top_k (int): Number of similar questions to retrieve

        Returns:
            str: The answer to the question
        """
        try:
            # Check if this is a conversational query instead of a health question
            conversational_type = self._is_conversational_query(question)
            if conversational_type == "greeting":
                logger.debug(f"Detected conversational query: '{question}'")
                return "I'm doing well, thanks for asking! How can I help you today?"
            elif conversational_type == "goodbye":
                return "Goodbye!  Redirecting to Google..." #Added redirect logic here (placeholder for actual redirect)

            # First check for exact matches (case-insensitive) to prioritize them
            normalized_question = question.lower().strip('?. ')

            # Check for exact match
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                # Check if this is an exact match
                if normalized_question == qa_normalized:
                    logger.debug(f"Found exact match for question: '{question}'")
                    logger.debug(f"Exact match index: {idx}")
                    return qa_pair['Answer']

            # Also check for questions that contain the exact query
            # This helps with cases like "what is the menstrual cycle" matching "what is the menstrual cycle?"
            for idx, qa_pair in enumerate(self.qa_pairs):
                qa_normalized = qa_pair['Question'].lower().strip('?. ')
                if qa_normalized.startswith(normalized_question) or normalized_question.startswith(qa_normalized):
                    logger.debug(f"Found partial match for question: '{question}'")
                    logger.debug(f"Partial match index: {idx}")
                    return qa_pair['Answer']

            # If no exact match, proceed with embedding-based retrieval
            # Generate embedding for the question
            question_embedding = self.generate_embeddings([question])

            # Search for similar questions
            distances, indices = self.index.search(question_embedding, top_k)

            # Perform a confidence check - don't answer if distance is too high
            if distances[0][0] > 15.0:
                logger.debug(f"Low confidence (distance: {distances[0][0]}) for query: '{question}'")
                return "I'm not sure I understand your question about reproductive health. Could you please rephrase it or ask something more specific about contraception, pregnancy, or reproductive health?"

            # If multiple good matches, combine answers
            if len(indices[0]) > 1 and distances[0][1] < 12.0:
                return self._combine_top_answers(question, distances[0], indices[0])

            # Get the most similar question's answer
            best_idx = indices[0][0]
            best_answer = self.qa_pairs[best_idx]['Answer']
            best_question = self.qa_pairs[best_idx]['Question']
            
            # Add citation from Planned Parenthood
            from chatbot.citation_manager import CitationManager
            citation_mgr = CitationManager()
            
            logger.debug(f"Primary confidence (distance): {distances[0][0]}")
            if len(indices[0]) > 1:
                logger.debug(f"Secondary confidence (gap): {distances[0][1] - distances[0][0]}")
            logger.debug(f"Matched question: {best_question}")
            
            # Add citation to the response
            cited_answer = citation_mgr.add_citation_to_text(best_answer, "planned_parenthood")
            return cited_answer

        except Exception as e:
            logger.error(f"Error getting RAG response: {str(e)}", exc_info=True)
            return "I apologize, but I encountered an error processing your question. Please try asking again or rephrase your question."
            return "I'm sorry, I couldn't find a good answer to your question."

    def _combine_top_answers(self, question, distances, indices, max_answers=3):
        """
        Combine the top answers for better response

        Args:
            question (str): The original question
            distances (list): Distances of top matches
            indices (list): Indices of top matches
            max_answers (int): Maximum number of answers to combine

        Returns:
            str: Combined answer
        """
        relevant_answers = []

        # Get the top answers that are within a reasonable distance
        for i in range(min(max_answers, len(indices))):
            if distances[i] < 15.0:  # Only include answers with reasonable similarity
                idx = indices[i]
                answer = self.qa_pairs[idx]['Answer']
                question_match = self.qa_pairs[idx]['Question']
                category = self.qa_pairs[idx].get('Category', 'General')

                relevant_answers.append({
                    'answer': answer,
                    'question': question_match,
                    'distance': distances[i],
                    'category': category
                })

        # If only one good match, just return it
        if len(relevant_answers) == 1:
            return relevant_answers[0]['answer']

        # Combine answers into a comprehensive response
        combined = "Based on your question, here's what I know:\n\n"

        # Group by category if available
        by_category = {}
        for item in relevant_answers:
            cat = item['category']
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(item)

        # Add information by category
        for category, items in by_category.items():
            if category != 'General':
                combined += f"## {category}\n"

            for item in items:
                # Extract the most important part of the answer
                answer_text = item['answer']
                if len(answer_text) > 300:
                    sentences = answer_text.split('. ')
                    if len(sentences) > 3:
                        answer_text = '. '.join(sentences[:3]) + '.'

                combined += f"{answer_text}\n\n"

        return combined

    def is_confident(self, question, response, threshold=8.0):
        """
        Determine if the RAG model is confident in its response
        Uses multiple metrics for better confidence assessment

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

            # Search for the closest questions
            distances, indices = self.index.search(question_embedding, 3)

            # Primary confidence: distance to closest match
            primary_confidence = distances[0][0]

            # Secondary confidence: gap between first and second match
            # A big gap means the first match is distinctly better
            secondary_confidence = 0
            if len(distances[0]) > 1:
                secondary_confidence = distances[0][1] - distances[0][0]

            # Get the matched question text for semantic similarity check
            matched_question = self.qa_pairs[indices[0][0]]['Question']

            # Log confidence metrics
            logger.debug(f"Primary confidence (distance): {primary_confidence}")
            logger.debug(f"Secondary confidence (gap): {secondary_confidence}")
            logger.debug(f"Matched question: {matched_question}")

            # Decision logic:
            # 1. If primary confidence is very good, trust it
            if primary_confidence < threshold * 0.7:
                logger.debug("High confidence based on primary score")
                return True

            # 2. If primary is okay and secondary shows a clear winner, trust it
            if primary_confidence < threshold and secondary_confidence > 2.0:
                logger.debug("Confidence based on primary score and distinct winner")
                return True

            # 3. If primary is just above threshold but very close to original question semantically, trust it
            if primary_confidence < threshold * 1.2 and self._is_semantically_similar(question, matched_question):
                logger.debug("Confidence based on semantic similarity")
                return True

            # Otherwise, not confident
            return False

        except Exception as e:
            logger.error(f"Error checking confidence: {str(e)}", exc_info=True)
            return False

    def _is_semantically_similar(self, q1, q2):
        """
        Check if two questions are semantically similar

        Args:
            q1 (str): First question
            q2 (str): Second question

        Returns:
            bool: True if semantically similar
        """
        # Simple keyword matching
        q1_words = set(q1.lower().split())
        q2_words = set(q2.lower().split())

        # Calculate Jaccard similarity
        intersection = len(q1_words.intersection(q2_words))
        union = len(q1_words.union(q2_words))

        if union == 0:
            return False

        similarity = intersection / union
        logger.debug(f"Semantic similarity: {similarity}")

        return similarity > 0.4