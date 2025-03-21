# Multi-Aspect Query Handling System

## Overview

The Multi-Aspect Query Handling System is a comprehensive architecture that enhances the Abby Chatbot's ability to process complex queries by breaking them down into multiple specialized aspects. This system provides more accurate, nuanced, and empathetic responses to users seeking information about reproductive health.

## Key Components

### 1. MultiAspectQueryProcessor

The central orchestrator that manages the entire query handling process. It coordinates classification, decomposition, specialized handling, and response composition.

### 2. UnifiedClassifier

Analyzes user queries and classifies them into three primary categories:
- Knowledge aspects (factual information)
- Emotional support aspects
- Policy/legal aspects

The classifier also identifies the confidence level for each aspect, detects sensitive topics, and recognizes location mentions.

### 3. AspectDecomposer

Breaks down complex, multi-faceted queries into separate aspects that can be handled by specialized components. It ensures that all dimensions of a user's question are addressed appropriately.

### 4. Specialized Handlers

- **KnowledgeHandler**: Processes factual questions about reproductive health using reliable information sources
- **EmotionalSupportHandler**: Provides empathetic and supportive responses for emotional aspects
- **PolicyHandler**: Handles legal/policy questions with location-specific context by integrating with the Abortion Policy API

### 5. ResponseComposer

Intelligently blends responses from different specialized handlers into coherent, well-structured answers. It uses appropriate transitions between aspects and ensures information isn't duplicated.

### 6. MemoryManager

Maintains conversation history and context to provide continuity and personalization across user interactions.

## How It Works

1. When a user sends a message, the `MultiAspectQueryProcessor` receives it and initiates the processing pipeline.

2. The `UnifiedClassifier` analyzes the message to determine its nature (knowledge, emotional, policy) and whether it contains multiple aspects.

3. The `AspectDecomposer` breaks the query into separate aspect-specific sub-queries if needed.

4. Each specialized handler processes its respective aspects in parallel.

5. The `ResponseComposer` combines all handler responses into a coherent final response.

6. The conversation history is updated by the `MemoryManager`.

## Benefits

- **Comprehensive Responses**: Addresses multiple dimensions of user queries in a single response
- **Specialized Expertise**: Each aspect is handled by a component designed specifically for that type of content
- **Emotional Intelligence**: Recognizes and addresses emotional needs alongside factual information
- **Location Awareness**: Provides policy information relevant to the user's location context
- **Efficient Processing**: Parallel handling of different aspects improves response time

## API Endpoints

- **POST /chat**: Process a message and generate a response
- **DELETE /session**: Clear a conversation session
- **GET /history/{session_id}**: Get conversation history for a session
- **POST /feedback**: Submit feedback for a message
- **GET /health**: Health check endpoint
- **GET /metrics**: Get performance metrics

## Getting Started

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Set up environment variables (create a `.env` file):
   ```
   OPENAI_API_KEY=your_openai_api_key
   POLICY_API_BASE_URL=https://api.abortionpolicyapi.com/v1/
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Access the chat interface at: http://localhost:8000

## Configuration Options

- **OpenAI Model**: You can configure which OpenAI model to use (default is "gpt-4o-mini")
- **Memory Size**: Configure the maximum history size (default is 20 messages)
- **API Integration**: Configure external API endpoints for policy data 