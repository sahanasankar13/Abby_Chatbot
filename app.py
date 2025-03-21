import logging
import os
import uuid
import time
import json
from typing import Dict, List, Any, Optional
import asyncio
import dotenv
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request, Depends, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Set up environment variables
dotenv.load_dotenv()

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our classes
from chatbot.multi_aspect_processor import MultiAspectQueryProcessor
from chatbot.memory_manager import MemoryManager

# Create FastAPI app
app = FastAPI(title="Abby Chatbot API", 
              description="Reproductive health chatbot with multi-aspect query handling",
              version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Configure templates
templates = Jinja2Templates(directory="templates")

# Initialize global components
memory_manager = MemoryManager()
query_processor = MultiAspectQueryProcessor()

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_location: Optional[Dict[str, str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    text: str
    message_id: str
    session_id: str
    citations: List[Any] = []
    citation_objects: List[Any] = []
    timestamp: float
    processing_time: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    graphics: Optional[List[Dict[str, Any]]] = None
    show_map: Optional[bool] = None
    zip_code: Optional[str] = None

class SessionRequest(BaseModel):
    session_id: str

class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str

class FeedbackRequest(BaseModel):
    message_id: str
    rating: int
    comment: Optional[str] = None

# Dependency to get the processor
def get_processor():
    return query_processor

# Dependency to get the memory manager
def get_memory_manager():
    return memory_manager

# Main page route
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Serve the main chat interface
    """
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request,
            "google_maps_api_key": os.getenv("GOOGLE_MAPS_API_KEY", "")
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, 
               processor: MultiAspectQueryProcessor = Depends(get_processor),
               memory: MemoryManager = Depends(get_memory_manager)):
    """
    Process a chat message and return a response
    """
    start_time = time.time()
    try:
        logger.info(f"Received chat request: {request.message[:100]}")
        
        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store user message in memory
        memory.add_message(
            session_id=session_id,
            message=request.message,
            role="user",
            metadata=request.metadata
        )
        
        # Get conversation history
        conversation_history = memory.get_history(session_id)
        
        # Process the query through our multi-aspect processor
        response_data = await processor.process_query(
            message=request.message,
            conversation_history=conversation_history,
            user_location=request.user_location
        )
        
        # Add message ID and session ID to response
        message_id = str(uuid.uuid4())
        response_data["message_id"] = message_id
        response_data["session_id"] = session_id
        
        if "processing_time" not in response_data:
            response_data["processing_time"] = time.time() - start_time
        
        # Store bot response in memory
        memory.add_message(
            session_id=session_id,
            message=response_data["text"],
            role="assistant",
            metadata={
                "message_id": message_id,
                "citations": response_data.get("citations", [])
            }
        )
        
        # Add empty graphics array for compatibility with frontend
        if "graphics" not in response_data:
            response_data["graphics"] = []
            
        # Extract state information from aspect_responses if available
        if "aspect_responses" in response_data:
            for aspect_response in response_data["aspect_responses"]:
                if aspect_response.get("aspect_type") == "policy":
                    # Copy state_code if it exists in this aspect response
                    if "state_code" in aspect_response and "state_code" not in response_data:
                        response_data["state_code"] = aspect_response["state_code"]
                        logger.info(f"Added state_code to response: {aspect_response['state_code']}")
                    
                    # Copy state_codes if it exists in this aspect response
                    if "state_codes" in aspect_response and "state_codes" not in response_data:
                        response_data["state_codes"] = aspect_response["state_codes"]
                        logger.info(f"Added state_codes to response: {aspect_response['state_codes']}")
                    
                    break
        
        # Log information about citations
        logger.info(f"Response has {len(response_data.get('citations', []))} citations")
        logger.info(f"Citation objects: {json.dumps(response_data.get('citation_objects', []))}")
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.delete("/session", status_code=200)
async def clear_session(request: SessionRequest, memory: MemoryManager = Depends(get_memory_manager)):
    """
    Clear a conversation session
    """
    try:
        result = memory.clear_session(request.session_id)
        if result:
            return {"status": "success", "message": f"Session {request.session_id} cleared"}
        else:
            return {"status": "warning", "message": f"Session {request.session_id} not found"}
            
    except Exception as e:
        logger.error(f"Error clearing session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")

@app.get("/history/{session_id}", status_code=200)
async def get_history(session_id: str, memory: MemoryManager = Depends(get_memory_manager)):
    """
    Get conversation history for a session
    """
    try:
        history = memory.get_history(session_id)
        return {"session_id": session_id, "messages": history}
            
    except Exception as e:
        logger.error(f"Error getting history: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting history: {str(e)}")

@app.post("/feedback", status_code=200)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit feedback for a message
    """
    try:
        message_id = request.message_id
        rating = request.rating
        comment = request.comment
        
        logger.info(f"Received feedback for message {message_id}: rating={rating}")
        
        # In a production system, store this feedback in a database
        # For now, just log it
        feedback_data = {
            "message_id": message_id,
            "rating": rating,
            "comment": comment,
            "timestamp": time.time()
        }
        
        # Append to a simple JSON file
        feedback_file = os.path.join(os.getcwd(), "user_feedback.json")
        try:
            # Read existing feedback
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r') as f:
                    try:
                        all_feedback = json.load(f)
                    except json.JSONDecodeError:
                        all_feedback = []
            else:
                all_feedback = []
                
            # Add new feedback
            all_feedback.append(feedback_data)
            
            # Write back to file
            with open(feedback_file, 'w') as f:
                json.dump(all_feedback, f, indent=2)
                
        except Exception as file_error:
            logger.error(f"Error saving feedback to file: {str(file_error)}")
        
        return {"success": True, "message": "Feedback recorded successfully"}
            
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check API health
    """
    return {
        "status": "ok",
        "version": app.version,
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/metrics", status_code=200)
async def get_metrics(processor: MultiAspectQueryProcessor = Depends(get_processor)):
    """
    Get performance metrics
    """
    try:
        metrics = processor.get_performance_metrics()
        return {"metrics": metrics}
            
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting metrics: {str(e)}")

@app.post("/test-multi-aspect", response_model=ChatResponse)
async def test_multi_aspect(request: ChatRequest, 
                         processor: MultiAspectQueryProcessor = Depends(get_processor),
                         memory: MemoryManager = Depends(get_memory_manager)):
    """
    Test endpoint specifically for processing multi-aspect queries
    This ensures the message is processed as a multi-aspect query even if the classifier might not detect it as such
    """
    try:
        start_time = time.time()
        logger.info(f"Received multi-aspect test request: {request.message[:50]}...")
        
        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        # Store user message in memory
        memory.add_message(
            session_id=session_id,
            message=request.message,
            role="user",
            metadata={"is_multi_aspect_test": True, **(request.metadata or {})}
        )
        
        # Get conversation history
        conversation_history = memory.get_history(session_id)
        
        # Force multi-aspect classification
        forced_classification = {
            "primary_type": "knowledge",
            "is_multi_aspect": True,
            "confidence_scores": {
                "knowledge": 0.7,
                "emotional": 0.6,
                "policy": 0.6
            },
            "topics": ["reproductive_health", "abortion", "state_laws", "emotional_support"],
            "sensitive_content": ["abortion"],
            "contains_location": True,
            "detected_locations": ["state"],
            "query_complexity": "complex"
        }
        
        # Process with forced multi-aspect
        async def process_with_forced_multi_aspect():
            # First classify the message normally
            classification = await processor.unified_classifier.classify(
                request.message, 
                conversation_history
            )
            
            # Override is_multi_aspect flag
            classification["is_multi_aspect"] = True
            
            # Boost all confidence scores to encourage multiple aspect processing
            for aspect_type in classification.get("confidence_scores", {}):
                classification["confidence_scores"][aspect_type] = max(
                    classification["confidence_scores"].get(aspect_type, 0), 
                    0.6
                )
                
            # Force query complexity to complex
            classification["query_complexity"] = "complex"
            
            # Decompose into aspects
            aspects = await processor.aspect_decomposer.decompose(
                request.message, 
                classification, 
                conversation_history
            )
            
            # Process each aspect
            aspect_responses = []
            aspect_tasks = []
            
            for aspect in aspects:
                aspect_type = aspect.get("type", "knowledge")
                
                if aspect_type in processor.handlers:
                    # Create processing task for this aspect
                    handler = processor.handlers[aspect_type]
                    task = asyncio.create_task(
                        processor._process_aspect(
                            handler=handler,
                            aspect=aspect,
                            message=request.message,
                            conversation_history=conversation_history,
                            user_location=request.user_location,
                            aspect_type=aspect_type
                        )
                    )
                    aspect_tasks.append(task)
            
            # Wait for all aspect processing to complete
            if aspect_tasks:
                aspect_responses = await asyncio.gather(*aspect_tasks)
                # Filter out None responses
                aspect_responses = [r for r in aspect_responses if r is not None]
            
            # Compose the final response
            response = processor.response_composer.compose_response(
                message=request.message,
                aspect_responses=aspect_responses,
                classification=classification
            )
            
            return response
        
        # Process the query with forced multi-aspect
        response_data = await process_with_forced_multi_aspect()
        
        # Add message ID and session ID to response
        message_id = str(uuid.uuid4())
        response_data["message_id"] = message_id
        response_data["session_id"] = session_id
        
        if "processing_time" not in response_data:
            response_data["processing_time"] = time.time() - start_time
        
        # Add debug info
        response_data["metadata"] = {
            "is_multi_aspect_test": True,
            "aspects_count": len(response_data.get("aspect_responses", [])) if "aspect_responses" in response_data else 0,
            **(response_data.get("metadata", {}))
        }
        
        # Store bot response in memory
        memory.add_message(
            session_id=session_id,
            message=response_data["text"],
            role="assistant",
            metadata={
                "message_id": message_id,
                "citations": response_data.get("citations", []),
                "is_multi_aspect_test": True
            }
        )
        
        # Add empty graphics array for compatibility with frontend
        if "graphics" not in response_data:
            response_data["graphics"] = []
            
        # Extract state information from aspect_responses if available
        if "aspect_responses" in response_data:
            for aspect_response in response_data["aspect_responses"]:
                if aspect_response.get("aspect_type") == "policy":
                    # Copy state_code if it exists in this aspect response
                    if "state_code" in aspect_response and "state_code" not in response_data:
                        response_data["state_code"] = aspect_response["state_code"]
                        logger.info(f"Added state_code to response: {aspect_response['state_code']}")
                    
                    # Copy state_codes if it exists in this aspect response
                    if "state_codes" in aspect_response and "state_codes" not in response_data:
                        response_data["state_codes"] = aspect_response["state_codes"]
                        logger.info(f"Added state_codes to response: {aspect_response['state_codes']}")
                    
                    break
            
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing multi-aspect test: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")

@app.post("/test-knowledge")
async def test_knowledge(request: ChatRequest):
    """
    Test endpoint to directly access the knowledge handler
    """
    try:
        logger.info(f"Testing knowledge handler with query: {request.message[:100]}")
        
        # Access the knowledge handler via the processor
        knowledge_handler = query_processor.handlers.get("knowledge")
        
        if not knowledge_handler:
            return {"error": "Knowledge handler not found"}
        
        # Process the query through the knowledge handler
        response = await knowledge_handler.process_query(
            query=request.message,
            full_message=request.message
        )
        
        # Log information about citations
        logger.info(f"Knowledge response has {len(response.get('citations', []))} citations")
        logger.info(f"Knowledge citation objects: {json.dumps(response.get('citation_objects', []))}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error testing knowledge handler: {str(e)}", exc_info=True)
        return {"error": f"Error testing knowledge handler: {str(e)}"}

@app.post("/test-citations")
async def test_citations(request: ChatRequest):
    """
    Test endpoint to directly create a response with citations
    """
    try:
        # Create a test response with explicit citation objects
        response = {
            "text": "Yes, it is normal to feel both relief and sadness after an abortion. Feeling relieved not to be pregnant and yet sad at the same time can be a confusing combination, but is common and understandable. Some women will know 'in their head' that having an abortion was the right decision and do not regret it, but at the same time feel sad 'in their heart' about the end of the pregnancy. However through time and talking with others this can resolve.",
            "citations": ["Planned Parenthood", "Sexual Health Sheffield"],
            "citation_objects": [
                {
                    "source": "Sexual Health Sheffield",
                    "url": "https://www.sexualhealthsheffield.nhs.uk/wp-content/uploads/2019/06/Wellbeing-after-an-abortion.pdf",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                },
                {
                    "source": "Planned Parenthood",
                    "url": "https://www.plannedparenthood.org/learn/abortion",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ],
            "confidence": 0.9,
            "aspect_type": "knowledge",
            "message_id": str(uuid.uuid4()),
            "processing_time": 0.1,
            "session_id": request.session_id or str(uuid.uuid4()),
            "timestamp": time.time()
        }
        
        # Add empty graphics array for compatibility with frontend
        if "graphics" not in response:
            response["graphics"] = []
            
        logger.info(f"Test citation response created with {len(response['citation_objects'])} citation objects")
        logger.info(f"Citation objects: {json.dumps(response['citation_objects'])}")
        
        return response
        
    except Exception as e:
        logger.error(f"Error creating test citations: {str(e)}", exc_info=True)
        return {"error": f"Error creating test citations: {str(e)}"}

@app.on_event("startup")
async def startup_event():
    """
    Initialize components on startup
    """
    logger.info("Starting up Abby Chatbot API")
    
    # In the future, we might want to initialize more components here
    # For example, loading pre-trained models, connecting to databases, etc.

@app.on_event("shutdown")
async def shutdown_event():
    """
    Clean up on shutdown
    """
    logger.info("Shutting down Abby Chatbot API")
    
    # Clean up inactive sessions
    session_count = memory_manager.cleanup_inactive_sessions()
    logger.info(f"Cleaned up {session_count} inactive sessions")

# Run the app
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=True)