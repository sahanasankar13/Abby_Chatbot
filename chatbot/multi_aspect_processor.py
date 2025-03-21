import logging
import asyncio
from typing import Dict, List, Any, Optional
import os
import time

from .unified_classifier import UnifiedClassifier
from .aspect_decomposer import AspectDecomposer
from .response_composer import ResponseComposer
from .knowledge_handler import KnowledgeHandler
from .emotional_support_handler import EmotionalSupportHandler
from .policy_handler import PolicyHandler

logger = logging.getLogger(__name__)

class MultiAspectQueryProcessor:
    """
    Main orchestrator for the multi-aspect query handling process.
    
    This class manages the full lifecycle of a user query:
    1. Classification
    2. Aspect decomposition
    3. Specialized handling per aspect
    4. Response composition
    
    It maintains references to all specialized handlers and manages their execution.
    """
    
    def __init__(self, 
                api_key: Optional[str] = None, 
                openai_model: str = "gpt-4o-mini",
                policy_api_base_url: Optional[str] = None):
        """
        Initialize the multi-aspect query processor
        
        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            openai_model (str): OpenAI model to use
            policy_api_base_url (Optional[str]): Base URL for the abortion policy API
        """
        logger.info("Initializing MultiAspectQueryProcessor")
        
        # Use provided API key or get from environment
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        # Use provided policy API URL or get from environment
        self.policy_api_base_url = policy_api_base_url or os.getenv("POLICY_API_BASE_URL", 
                                                                   "https://api.abortionpolicyapi.com/v1/")
        
        # Initialize components
        self.unified_classifier = UnifiedClassifier(api_key=self.api_key, model_name=openai_model)
        self.aspect_decomposer = AspectDecomposer(api_key=self.api_key, model_name=openai_model)
        self.response_composer = ResponseComposer()
        
        # Initialize specialized handlers
        self.handlers = {
            "knowledge": KnowledgeHandler(api_key=self.api_key, model_name=openai_model),
            "emotional": EmotionalSupportHandler(api_key=self.api_key, model_name=openai_model),
            "policy": PolicyHandler(api_key=self.api_key, policy_api_base_url=self.policy_api_base_url)
        }
        
        # Performance tracking
        self.processing_times = {
            "classification": [],
            "decomposition": [],
            "handling": {},
            "composition": []
        }
        for handler_type in self.handlers.keys():
            self.processing_times["handling"][handler_type] = []
    
    async def process_query(self, 
                          message: str, 
                          conversation_history: List[Dict[str, Any]] = None,
                          user_location: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a user query through the multi-aspect pipeline
        
        Args:
            message (str): User query text
            conversation_history (List[Dict[str, Any]]): Previous conversation messages
            user_location (Optional[Dict[str, str]]): User's location data (state, city, etc.)
            
        Returns:
            Dict[str, Any]: Processed response
        """
        start_time = time.time()
        logger.info(f"Processing query: {message[:100]}...")
        
        try:
            # Initialize conversation history if None
            if conversation_history is None:
                conversation_history = []
            
            # 1. Classify the query
            classification_start = time.time()
            classification = await self.unified_classifier.classify(
                message, 
                conversation_history
            )
            self.processing_times["classification"].append(time.time() - classification_start)
            
            logger.info(f"Classification result: {classification}")
            
            # 2. Decompose into aspects if needed
            decomposition_start = time.time()
            aspects = await self.aspect_decomposer.decompose(
                message, 
                classification, 
                conversation_history
            )
            self.processing_times["decomposition"].append(time.time() - decomposition_start)
            
            logger.info(f"Decomposed into {len(aspects)} aspects")
            
            # 3. Process each aspect with specialized handlers
            aspect_responses = []
            aspect_tasks = []
            
            for aspect in aspects:
                aspect_type = aspect.get("type", "knowledge")
                
                if aspect_type in self.handlers:
                    # Create processing task for this aspect
                    handler = self.handlers[aspect_type]
                    task = asyncio.create_task(
                        self._process_aspect(
                            handler=handler,
                            aspect=aspect,
                            message=message,
                            conversation_history=conversation_history,
                            user_location=user_location,
                            aspect_type=aspect_type
                        )
                    )
                    aspect_tasks.append(task)
                else:
                    logger.warning(f"No handler available for aspect type: {aspect_type}")
            
            # Wait for all aspect processing to complete
            if aspect_tasks:
                aspect_responses = await asyncio.gather(*aspect_tasks)
                # Filter out None responses
                aspect_responses = [r for r in aspect_responses if r is not None]
            
            # 4. Compose the final response
            composition_start = time.time()
            response = self.response_composer.compose_response(
                message=message,
                aspect_responses=aspect_responses,
                classification=classification
            )
            self.processing_times["composition"].append(time.time() - composition_start)
            
            # Add processing time to the response
            response["processing_time"] = time.time() - start_time
            
            # Ensure state information is included in the main response
            for aspect_response in aspect_responses:
                # Copy state_code if present
                if "state_code" in aspect_response and "state_code" not in response:
                    response["state_code"] = aspect_response["state_code"]
                    logger.info(f"Added state_code to main response: {aspect_response['state_code']}")
                
                # Copy state_codes if present
                if "state_codes" in aspect_response and "state_codes" not in response:
                    response["state_codes"] = aspect_response["state_codes"]
                    logger.info(f"Added state_codes to main response: {aspect_response['state_codes']}")
            
            # Store aspect responses for debugging/metrics
            response["aspect_responses"] = aspect_responses
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            # Return fallback response
            return {
                "text": "I apologize, but I encountered an error processing your question. Could you try rephrasing or asking a different question?",
                "citations": [],
                "citation_objects": [],
                "processing_time": time.time() - start_time
            }
    
    async def _process_aspect(self, 
                             handler: Any, 
                             aspect: Dict[str, Any],
                             message: str,
                             conversation_history: List[Dict[str, Any]],
                             user_location: Optional[Dict[str, str]],
                             aspect_type: str) -> Optional[Dict[str, Any]]:
        """
        Process a single aspect using the appropriate handler
        
        Args:
            handler: The specialized handler
            aspect: The aspect data
            message: Original user message
            conversation_history: Previous conversation messages
            user_location: User's location data
            aspect_type: Type of the aspect
            
        Returns:
            Optional[Dict[str, Any]]: Handler response or None if failed
        """
        try:
            start_time = time.time()
            
            # Extract the aspect query
            aspect_query = aspect.get("query", message)
            confidence = aspect.get("confidence", 0.8)
            
            # Process query with specialized handler
            response = await handler.process_query(
                query=aspect_query,
                full_message=message,
                conversation_history=conversation_history,
                user_location=user_location
            )
            
            # If response is valid, add metadata
            if response and isinstance(response, dict):
                response["aspect_type"] = aspect_type
                response["confidence"] = confidence
                
                # Track processing time
                elapsed_time = time.time() - start_time
                self.processing_times["handling"][aspect_type].append(elapsed_time)
                
                return response
            else:
                logger.warning(f"Handler for {aspect_type} returned invalid response")
                return None
                
        except Exception as e:
            logger.error(f"Error processing aspect {aspect_type}: {str(e)}", exc_info=True)
            return None
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for the query processor
        
        Returns:
            Dict[str, Any]: Performance metrics
        """
        metrics = {
            "average_times": {},
            "total_queries_processed": len(self.processing_times["classification"]),
            "handler_usage": {}
        }
        
        # Calculate average times
        if self.processing_times["classification"]:
            metrics["average_times"]["classification"] = sum(self.processing_times["classification"]) / len(self.processing_times["classification"])
        
        if self.processing_times["decomposition"]:
            metrics["average_times"]["decomposition"] = sum(self.processing_times["decomposition"]) / len(self.processing_times["decomposition"])
        
        if self.processing_times["composition"]:
            metrics["average_times"]["composition"] = sum(self.processing_times["composition"]) / len(self.processing_times["composition"])
        
        # Calculate average times and usage for each handler
        for handler_type, times in self.processing_times["handling"].items():
            if times:
                metrics["average_times"][f"handling_{handler_type}"] = sum(times) / len(times)
                metrics["handler_usage"][handler_type] = len(times)
            else:
                metrics["handler_usage"][handler_type] = 0
        
        return metrics 