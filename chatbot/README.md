# Reproductive Health Chatbot Components

This directory contains the core components of the reproductive health chatbot system, organized into modular files that work together to provide accurate, empathetic responses on reproductive health topics.

## File Descriptions

### `__init__.py`
- Initializes the chatbot package, making components available for import

### `baseline_model.py`
- Implements the core `BaselineModel` class that orchestrates different response modes
- Categorizes questions into policy, knowledge, and conversational types
- Coordinates between RAG-based knowledge, policy API data, and empathetic responses
- Handles multi-query questions by splitting them and processing each part

### `bert_rag.py`
- Implements Retrieval-Augmented Generation (RAG) using BERT embeddings
- Performs semantic search on a database of 550+ reproductive health Q&A pairs
- Features advanced relevance scoring and confidence assessment 
- Provides natural language response formatting for improved readability
- Handles out-of-scope questions and conversational queries

### `citation_manager.py`
- Manages attribution of information sources in responses
- Tracks citations from Planned Parenthood and Abortion Policy API
- Formats citations for display in the user interface
- Ensures transparency about information sources

### `conversation_manager.py`
- Manages the overall conversation flow between user and chatbot
- Detects location context for policy-related questions
- Combines emotional support with factual information
- Handles complex queries that combine multiple aspects (policy, emotional, educational)
- Maintains conversation history for context-aware responses

### `friendly_bot.py`
- Adds empathetic, supportive elements to technical responses
- Improves response structure and readability
- Detects different question types to provide appropriate emotional tone
- Ensures responses are both accurate and emotionally supportive

### `gpt_integration.py`
- Integrates with OpenAI's GPT models for enhanced conversational responses
- Provides fallback for questions outside the RAG model's knowledge
- Enhances response quality for complex or ambiguous questions
- Formats policy data into natural, user-friendly language

### `policy_api.py`
- Integrates with the Abortion Policy API for up-to-date state-specific information
- Extracts state context from user queries
- Formats complex policy data into clear, accurate responses
- Handles cases where state information is unavailable or unclear

### `response_evaluator.py`
- Evaluates response quality before delivery to users
- Uses GPT-4 or alternative models to assess relevance and completeness
- Flags potential issues in responses for refinement
- Ensures responses directly answer the user's questions

### `visual_info.py`
- Provides visual information graphics for reproductive health topics
- Creates SVG-based visualizations for topics like menstrual cycles
- Suggests relevant graphics based on conversation content
- Enhances educational value through visual explanations

## Architecture Overview

The chatbot uses a tiered approach to handling questions:

1. **Query Analysis**: Determines if the question is about policy, needs factual information, or is conversational
2. **Response Generation**: Uses the appropriate model based on the query type
3. **Response Enhancement**: Adds emotional support and friendly elements
4. **Quality Assurance**: Evaluates response quality before delivery
5. **Citation Management**: Adds proper attribution to information sources

This modular architecture allows for easy maintenance and extension of the system's capabilities.