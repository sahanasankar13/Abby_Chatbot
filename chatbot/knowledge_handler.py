import logging
import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
import re
import time
from typing import Dict, List, Any, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from datetime import datetime
import json
from openai import OpenAI

# Import for OpenAI
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

logger = logging.getLogger(__name__)

class KnowledgeHandler:
    """
    Handler for knowledge-seeking aspects of user queries.
    
    This class processes factual questions about reproductive health,
    accessing reliable knowledge sources and generating informative responses.
    """
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4o-mini"):
        """
        Initialize the knowledge handler
        
        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            model_name (str): OpenAI model to use
        """
        logger.info(f"Initializing KnowledgeHandler with model {model_name}")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Set up OpenAI client
        self.client = OpenAI(api_key=self.api_key)
        self.model = model_name
        
        # Initialize data structures
        self.data = pd.DataFrame()
        self.index = None
        self.embeddings = None
        
        # Try to load BERT model for embeddings
        try:
            from transformers import AutoModel, AutoTokenizer
            self.embedding_model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Successfully loaded BERT model for embeddings")
            
            # Load datasets and build index
            self.data = self._load_datasets()
            self.index, self.embeddings = self._build_index()
        except Exception as e:
            logger.error(f"Error loading BERT model: {str(e)}")
            self.embedding_model = None
            self.tokenizer = None
        
        # Knowledge response prompt template
        self.knowledge_prompt = """You are a knowledgeable and compassionate reproductive health specialist providing factual information.

User query: {query}

Full message context: {full_message}

Using the following knowledge sources to inform your response:
{knowledge_sources}

Respond with accurate, evidence-based information about reproductive health. Be compassionate but focus on facts. 
Clearly cite your sources. If you're uncertain about information, acknowledge the limits of your knowledge.
If the question is outside your expertise or knowledge sources, say so and suggest consulting a healthcare provider.

Keep your response focused on reproductive health information and avoid any political statements.
Make your response concise but comprehensive.
"""
    
    async def process_query(self, query: str, full_message: Optional[str] = None, 
                          conversation_history: Optional[List[Dict[str, Any]]] = None,
                          user_location: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a knowledge query and return a response
        
        Args:
            query (str): The knowledge-specific query aspect
            full_message (str): The original complete user message
            conversation_history (List[Dict[str, Any]]): Previous conversation messages
            user_location (Optional[Dict[str, str]]): User's location data (not used by knowledge handler)
            
        Returns:
            Dict[str, Any]: Response with text and citations
        """
        try:
            start_time = time.time()
            logger.info(f"Processing knowledge query: {query[:100]}...")
            
            # Use the provided query or full message if query is None
            query_text = query or full_message
            if not query_text:
                raise ValueError("No query text provided")
            
            # Use vector search to find relevant documents
            if self.index is not None:
                docs, scores = await self._retrieve_context(query_text, top_k=5)
                
                if docs and len(docs) > 0:
                    logger.info(f"Found {len(docs)} relevant documents from vector search")
                    
                    # Generate response based on retrieved documents
                    if self.client:
                        # Format the documents for the prompt
                        formatted_sources = self._format_vector_sources(docs)
                        
                        # Generate response with OpenAI
                        prompt = self.knowledge_prompt.format(
                            query=query_text,
                            full_message=full_message or query_text,
                            knowledge_sources=formatted_sources
                        )
                        
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "system", "content": prompt}],
                            temperature=0.3
                        )
                        
                        # Extract the response text
                        response_text = response.choices[0].message.content
                    else:
                        # Simple response using the most relevant document if no OpenAI
                        response_text = self._format_simple_response(docs)
                    
                    # Create citation objects directly from the documents
                    citation_objects = []
                    for i, doc in enumerate(docs):
                        if i >= 3:  # Limit to the top 3 most relevant sources
                            break
                            
                        source = doc.get("source", "Unknown Source")
                        url = doc.get("url", "")
                        
                        # Only add citations with URLs
                        if url:
                            # Extract a meaningful title from the URL or use the document title if available
                            title = doc.get("title", "")
                            if not title and url:
                                try:
                                    from urllib.parse import urlparse
                                    parsed_url = urlparse(url)
                                    path_parts = parsed_url.path.strip('/').split('/')
                                    if path_parts and path_parts[-1]:
                                        # Convert last path segment to a readable title
                                        raw_title = path_parts[-1].replace('-', ' ').replace('_', ' ')
                                        title = ' '.join(word.capitalize() for word in raw_title.split())
                                    else:
                                        title = source
                                except Exception as e:
                                    logger.warning(f"Error extracting title from URL: {str(e)}")
                                    title = source
                            
                            citation_obj = {
                                "source": source,
                                "url": url,
                                "title": title or source,
                                "accessed_date": datetime.now().strftime('%Y-%m-%d')
                            }
                            citation_objects.append(citation_obj)
                    
                    # Create simple citation sources for backward compatibility
                    citation_sources = [c["source"] for c in citation_objects]
                    
                    # Log the citations with details
                    logger.info(f"Found {len(citation_objects)} citations with URLs:")
                    for c in citation_objects:
                        logger.info(f"  - {c['source']}: {c['url']}")
                    
                    # Measure processing time
                    processing_time = time.time() - start_time
                    logger.info(f"Knowledge response generated in {processing_time:.2f} seconds")
                    
                    # Return the result
                    return {
                        "text": response_text,
                        "citations": citation_sources,
                        "citation_objects": citation_objects,
                        "aspect_type": "knowledge",
                        "confidence": 0.8,
                        "processing_time": processing_time
                    }
            
            # Fallback response if no relevant documents found
            logger.warning("No relevant documents found for query")
            return {
                "text": "I don't have specific information to answer your question about reproductive health. Consider contacting a healthcare provider for personalized guidance.",
                "citations": [],
                "citation_objects": [],
                "aspect_type": "knowledge",
                "confidence": 0.3,
                "processing_time": time.time() - start_time
            }
            
        except Exception as e:
            logger.error(f"Error processing knowledge query: {str(e)}", exc_info=True)
            return {
                "text": "I apologize, but I couldn't find specific information to answer your question about reproductive health. Could you try rephrasing your question or ask about a different topic?",
                "citations": [],
                "citation_objects": [],
                "aspect_type": "knowledge",
                "confidence": 0.2
            }
    
    def _find_relevant_sources(self, query: str) -> List[Dict[str, Any]]:
        """
        Find relevant knowledge sources for a query
        
        Args:
            query (str): The user query
            
        Returns:
            List[Dict[str, Any]]: List of relevant knowledge sources
        """
        # This method is no longer used - all queries go through vector search
        return []
    
    def _format_knowledge_sources(self, sources: List[Dict[str, Any]]) -> str:
        """
        Format knowledge sources for inclusion in the prompt
        
        Args:
            sources (List[Dict[str, Any]]): Knowledge sources to format
            
        Returns:
            str: Formatted knowledge sources text
        """
        if not sources:
            return "No specific knowledge sources available for this query."
        
        formatted_text = ""
        for source in sources:
            formatted_text += f"SOURCE: {source.get('source', 'Unknown Source')}\n"
            formatted_text += f"{source.get('answer', '')}\n\n"
        
        return formatted_text
        
    def _format_vector_sources(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format vector search results for inclusion in the prompt
        
        Args:
            docs (List[Dict[str, Any]]): Retrieved documents
            
        Returns:
            str: Formatted knowledge sources text
        """
        if not docs:
            return "No specific knowledge sources available for this query."
        
        formatted_text = ""
        for i, doc in enumerate(docs):
            formatted_text += f"SOURCE [{i+1}]: {doc.get('source', 'Unknown Source')}\n"
            formatted_text += f"Question: {doc.get('question', '')}\n"
            formatted_text += f"Answer: {doc.get('answer', '')}\n\n"
        
        return formatted_text

    def _load_datasets(self) -> pd.DataFrame:
        """
        Load and prepare knowledge datasets
        
        Returns:
            pd.DataFrame: Combined dataset with questions, answers and metadata
        """
        try:
            # Load abortion FAQ dataset
            df_abortion = pd.read_csv("data/AbortionPPDFAQ.csv", skiprows=1)  # Skip the header row
            
            # Ensure column names are correct for the abortion dataset
            if "question" in df_abortion.columns and "answer" in df_abortion.columns:
                df_abortion = df_abortion.rename(columns={
                    "question": "Question",
                    "answer": "Answer"
                })
            
            # Handle Link column in abortion dataset
            if "Link" in df_abortion.columns and "URL" not in df_abortion.columns:
                df_abortion = df_abortion.rename(columns={"Link": "URL"})
            
            # Add source if not present
            if "Source" not in df_abortion.columns:
                df_abortion["Source"] = "Planned Parenthood Abortion FAQ"
            
            # Load general reproductive health dataset
            df_general = pd.read_csv("data/Planned Parenthood Data - Sahana.csv")
            
            # Ensure consistent column names for the general dataset
            if "question" in df_general.columns and "answer" in df_general.columns:
                df_general = df_general.rename(columns={
                    "question": "Question",
                    "answer": "Answer"
                })
            elif "Title" in df_general.columns and "Content" in df_general.columns:
                df_general = df_general.rename(columns={
                    "Title": "Question", 
                    "Content": "Answer"
                })
            
            # Handle the Link column in general dataset
            if "Link" in df_general.columns and "URL" not in df_general.columns:
                df_general = df_general.rename(columns={"Link": "URL"})
            
            # Add source if not present
            if "Source" not in df_general.columns:
                df_general["Source"] = "Planned Parenthood"
            
            # Ensure URL exists in both datasets
            if "URL" not in df_general.columns:
                df_general["URL"] = "https://www.plannedparenthood.org/learn"
            
            if "URL" not in df_abortion.columns:
                df_abortion["URL"] = "https://www.plannedparenthood.org/learn/abortion"
            
            # Combine datasets
            df_combined = pd.concat([df_abortion, df_general], ignore_index=True)
            
            # Clean data
            df_combined = df_combined.dropna(subset=["Question", "Answer"])
            df_combined["Question"] = df_combined["Question"].astype(str)
            df_combined["Answer"] = df_combined["Answer"].astype(str)
            df_combined["URL"] = df_combined["URL"].astype(str)  # Ensure URL is string
            
            logger.info(f"Loaded dataset with {len(df_combined)} entries")
            return df_combined
            
        except Exception as e:
            logger.error(f"Error loading datasets: {str(e)}", exc_info=True)
            # Return an empty dataframe with the required columns
            return pd.DataFrame(columns=["Question", "Answer", "Source", "URL"])
    
    def _build_index(self) -> Tuple[faiss.Index, np.ndarray]:
        """
        Build FAISS index for efficient similarity search
        
        Returns:
            Tuple[faiss.Index, np.ndarray]: FAISS index and document embeddings
        """
        try:
            if self.embedding_model is None or len(self.data) == 0:
                logger.warning("Cannot build index: model or data not available")
                return None, np.array([])
            
            # Create a combined text field for indexing (question + answer)
            texts = self.data["Question"].tolist()
            
            # Generate embeddings
            embeddings = self._generate_embeddings(texts)
            
            if embeddings.size == 0:
                logger.warning("Generated empty embeddings, cannot build index")
                return None, np.array([])
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
            
            logger.info(f"Built FAISS index with {len(texts)} documents and dimension {dimension}")
            return index, embeddings
            
        except Exception as e:
            logger.error(f"Error building index: {str(e)}", exc_info=True)
            return None, np.array([])
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of text strings to embed
            
        Returns:
            np.ndarray: Embeddings array
        """
        try:
            if self.embedding_model is None:
                logger.warning("Model not available for generating embeddings")
                return np.array([])
            
            embeddings = []
            batch_size = 32
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize and get embeddings
                inputs = self.tokenizer(batch_texts, padding=True, truncation=True, 
                                       return_tensors="pt", max_length=512)
                
                with torch.no_grad():
                    outputs = self.embedding_model(**inputs)
                
                # Use mean pooling to get sentence embeddings
                attention_mask = inputs['attention_mask']
                mean_embeddings = self._mean_pooling(outputs.last_hidden_state, attention_mask)
                
                # Normalize
                normalized_embeddings = torch.nn.functional.normalize(mean_embeddings, p=2, dim=1)
                embeddings.append(normalized_embeddings.numpy())
            
            return np.vstack(embeddings) if embeddings else np.array([])
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
            return np.array([])
    
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
    
    async def _retrieve_context(self, query: str, top_k: int = 5) -> Tuple[List[Dict[str, Any]], List[float]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query (str): The query to search for
            top_k (int): Number of documents to retrieve
            
        Returns:
            Tuple[List[Dict[str, Any]], List[float]]: Retrieved documents and their similarity scores
        """
        try:
            if self.index is None or len(self.data) == 0:
                logger.warning("Cannot retrieve context: index or data not available")
                return [], []
            
            # Generate query embedding
            query_embedding = self._generate_embeddings([query])
            
            if query_embedding.size == 0:
                logger.warning("Generated empty query embedding, cannot search")
                return [], []
            
            # Search the index
            k = min(top_k, len(self.data))
            distances, indices = self.index.search(query_embedding, k)
            
            # Convert to document objects
            documents = []
            scores = distances[0].tolist()  # Convert to Python list
            
            for i, idx in enumerate(indices[0]):
                if idx >= 0 and idx < len(self.data):  # Check if index is valid
                    row = self.data.iloc[idx]
                    document = {
                        "question": row.get("Question", ""),
                        "answer": row.get("Answer", ""),
                        "source": row.get("Source", "Unknown Source"),
                    }
                    
                    # Check for URL in both URL and Link columns
                    url = row.get("URL", "")
                    if not url and "Link" in row:
                        url = row.get("Link", "")
                    document["url"] = url
                    
                    # Add similarity score
                    document["score"] = float(scores[i])
                    documents.append(document)
            
            logger.info(f"Retrieved {len(documents)} documents for query: {query[:50]}...")
            return documents, scores
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
            return [], []
    
    async def _generate_with_openai(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using OpenAI with retrieved context
        
        Args:
            query (str): The user's query
            docs (List[Dict[str, Any]]): Retrieved documents
            
        Returns:
            str: Generated response
        """
        if not self.client:
            raise ValueError("OpenAI client not available")
        
        # Prepare context from retrieved documents
        context = ""
        for i, doc in enumerate(docs):
            context += f"Document {i+1}:\n"
            context += f"Question: {doc['question']}\n"
            context += f"Answer: {doc['answer']}\n"
            context += f"Source: {doc['source']}\n\n"
        
        system_message = """You are an expert reproductive health assistant specialized in providing accurate information.
Use the provided context to answer the user's question about reproductive health.
If the context doesn't contain the necessary information to fully answer the question, acknowledge that
and provide what information you do have from the context.
Your response should be concise, accurate, and helpful. Format citations in your answer as [1], [2], etc.
IMPORTANT: Only use information from the provided context. Don't make up information or cite sources not in the context."""
        
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o",  # Use the most advanced model for best quality
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating response with OpenAI: {str(e)}", exc_info=True)
            # Fall back to simple retrieval
            return self._combine_retrieved_docs(docs, [0.8] * len(docs))
    
    def _combine_retrieved_docs(self, docs: List[Dict[str, Any]], 
                               scores: List[float]) -> str:
        """
        Combine retrieved documents into a coherent response
        
        Args:
            docs (List[Dict[str, Any]]): Retrieved documents
            scores (List[float]): Similarity scores for the documents
            
        Returns:
            str: Combined response
        """
        if not docs:
            return "I don't have specific information about that. Could you ask another question about reproductive health?"
        
        # Use the most relevant document as the primary response
        primary_doc = docs[0]
        primary_answer = primary_doc['answer']
        
        # For very short responses, add information from other documents
        if len(primary_answer.split()) < 30 and len(docs) > 1:
            additional_info = []
            for i, doc in enumerate(docs[1:3]):  # Use up to 2 additional docs
                # Only add if it adds new information
                if not self._is_similar_to(doc['answer'], primary_answer, 0.7):
                    additional_info.append(doc['answer'])
            
            if additional_info:
                combined_response = primary_answer + " " + " ".join(additional_info)
                return combined_response
        
        return primary_answer
    
    def _is_similar_to(self, text1: str, text2: str, threshold: float = 0.7) -> bool:
        """
        Check if two texts are similar
        
        Args:
            text1 (str): First text
            text2 (str): Second text
            threshold (float): Similarity threshold
            
        Returns:
            bool: True if texts are similar
        """
        # Simple word overlap for similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard_sim = len(intersection) / len(union)
        return jaccard_sim > threshold
    
    def _extract_citations(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Extract formatted citations from documents
        
        Args:
            docs (List[Dict[str, Any]]): Retrieved documents
            
        Returns:
            List[Dict[str, Any]]: Formatted citations
        """
        unique_citations = {}
        
        for i, doc in enumerate(docs):
            source = doc.get('source', 'Unknown Source')
            
            # Check for URL in both lowercase and uppercase keys
            url = doc.get('url', '')
            if not url and 'URL' in doc:
                url = doc.get('URL', '')
            
            # Create a unique key to avoid duplicate citations
            key = f"{source}_{url}"
            
            if key not in unique_citations:
                citation_obj = {
                    "id": i + 1,
                    "source": source,
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
                
                # Add the URL if available
                if url:
                    citation_obj["url"] = url
                    
                # Add the text snippet if available
                if "answer" in doc:
                    text_snippet = doc["answer"]
                    max_len = 100
                    if len(text_snippet) > max_len:
                        text_snippet = text_snippet[:max_len] + "..."
                    citation_obj["text"] = text_snippet
                
                unique_citations[key] = citation_obj
        
        return list(unique_citations.values()) 