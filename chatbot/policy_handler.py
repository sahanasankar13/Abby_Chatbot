import logging
import os
import asyncio
import aiohttp
import json
import time
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import for OpenAI
try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None

# Import for ZIP code lookup - optional, will use fallback if not available
try:
    from pyzipcode import ZipCodeDatabase
except ImportError:
    ZipCodeDatabase = None

logger = logging.getLogger(__name__)

class PolicyHandler:
    """
    Handles policy-related queries with location context awareness
    """
    
    def __init__(self, api_key: Optional[str] = None, policy_api_base_url: Optional[str] = None):
        """
        Initialize the policy handler
        
        Args:
            api_key (Optional[str]): OpenAI API key, defaults to environment variable
            policy_api_base_url (Optional[str]): Base URL for the abortion policy API
        """
        logger.info("Initializing PolicyHandler")
        
        # Load environment variables or use provided values
        self.openai_api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.abortion_policy_api_key = os.environ.get("ABORTION_POLICY_API_KEY", "")  # Default empty string
        self.policy_api_base_url = policy_api_base_url or os.environ.get("POLICY_API_BASE_URL", "https://api.abortionpolicyapi.com/v1/")
        self.openai_client = None
        
        if self.openai_api_key and AsyncOpenAI:
            self.openai_client = AsyncOpenAI(api_key=self.openai_api_key)
            logger.info("OpenAI client initialized for policy handler")
        else:
            logger.warning("OpenAI client not available, using template-based fallback")
        
        # Define US state codes and names
        self.STATE_NAMES = {
            'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas',
            'CA': 'California', 'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware',
            'FL': 'Florida', 'GA': 'Georgia', 'HI': 'Hawaii', 'ID': 'Idaho',
            'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa', 'KS': 'Kansas',
            'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
            'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
            'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada',
            'NH': 'New Hampshire', 'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York',
            'NC': 'North Carolina', 'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma',
            'OR': 'Oregon', 'PA': 'Pennsylvania', 'RI': 'Rhode Island', 'SC': 'South Carolina',
            'SD': 'South Dakota', 'TN': 'Tennessee', 'TX': 'Texas', 'UT': 'Utah',
            'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington', 'WV': 'West Virginia',
            'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'District of Columbia'
        }
        
        # Create lowercase state names for searches
        self.STATE_NAMES_LOWER = {k.lower(): k for k in self.STATE_NAMES.keys()}
        for k, v in self.STATE_NAMES.items():
            self.STATE_NAMES_LOWER[v.lower()] = k

        # Initialize cache for policy API responses
        self.policy_cache = {}
        self.cache_ttl = 86400  # 24 hours - policy data doesn't change often
        
        # Keep track of state associations for sessions
        self.session_state_cache = {}  # Maps session_id -> state_code
        
        # Endpoints for the Abortion Policy API
        self.api_base_url = self.policy_api_base_url
        self.api_endpoints = {
            "waiting_periods": "waiting_periods",
            "insurance_coverage": "insurance_coverage",
            "gestational_limits": "gestational_limits",
            "minors": "minors"
        }
    
    async def process_query(self, 
                          query: str, 
                          full_message: str = None,
                          conversation_history: List[Dict[str, Any]] = None,
                          user_location: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Process a policy query, extracting location context when available
        
        Args:
            query (str): The specific policy query aspect
            full_message (str): The original complete user message
            conversation_history (List[Dict[str, Any]]): Previous conversation messages
            user_location (Optional[Dict[str, str]]): User's location data
            
        Returns:
            Dict[str, Any]: Response with policy information
        """
        try:
            start_time = time.time()
            logger.info(f"Processing policy query: '{query[:100]}'...")
            
            # Extract session ID from conversation history if available
            session_id = None
            if conversation_history and len(conversation_history) > 0:
                first_msg = conversation_history[0]
                if "session_id" in first_msg:
                    session_id = first_msg["session_id"]
                    logger.info(f"Found session ID: {session_id}")
                    
                    # Debug: if we have a session ID, check if there's a cached state
                    if session_id in self.session_state_cache:
                        logger.info(f"Session {session_id} has cached state: {self.session_state_cache[session_id]}")
                    else:
                        logger.info(f"Session {session_id} has no cached state yet")
            
            # Check for comparison between states (example: "What are the abortion laws in Texas vs California?")
            comparing_states = False
            compare_pattern = re.search(r'\b(vs\.?|versus|compared to|compare with|difference between)\b', 
                                       full_message.lower() if full_message else query.lower())
            if compare_pattern:
                comparing_states = True
                logger.info("Detected comparison query between multiple states")
            
            # Check for multiple state mentions
            logger.info(f"Checking for state mentions in query: '{query}'")
            if full_message and full_message != query:
                logger.info(f"Also checking full message: '{full_message}'")
                
            # Find all state mentions in the query
            state_mentions = self._get_all_state_mentions(query=query, full_message=full_message or query)
            
            if len(state_mentions) > 1:
                logger.info(f"Multiple states detected: {', '.join(state_mentions)}")
                
                # If comparing states, handle as a special case
                if comparing_states:
                    return await self._handle_state_comparison(query, state_mentions, full_message)
                
                # Otherwise, default to using the first mentioned state but note the others
                state_code = state_mentions[0]
                
                # Update the session cache with the first state
                if session_id:
                    self.session_state_cache[session_id] = state_code
                    logger.info(f"✓ Updated session cache with first mentioned state: {session_id} -> {state_code}")
                
                # Add a note that we're only using the first state
                note_about_multiple = f"Note: I've detected mentions of multiple states ({', '.join([self.STATE_NAMES.get(s, s) for s in state_mentions])}). I'll focus on {self.STATE_NAMES.get(state_code, state_code)} first, but can provide information about the others if you'd like."
                
            elif len(state_mentions) == 1:
                # Single state explicitly mentioned
                state_code = state_mentions[0]
                logger.info(f"✓ Found single state {state_code} ({self.STATE_NAMES.get(state_code, '')}) in current message")
                
                # Update the session cache with the state
                if session_id:
                    self.session_state_cache[session_id] = state_code
                    logger.info(f"✓ Updated session cache: {session_id} -> {state_code}")
                
                note_about_multiple = ""
            else:
                # No state found in current query
                state_code = None
                note_about_multiple = ""
                
                # Check if this is a "my state" query
                asking_about_my_state = False
                if re.search(r'\b(my|our)\s+state\b', query.lower()) or re.search(r'\b(my|our)\s+state\b', full_message.lower() if full_message else ''):
                    asking_about_my_state = True
                    logger.info("User is asking about 'my state'")
                    
                    # Check if we have a cached state for this session
                    if session_id and session_id in self.session_state_cache:
                        cached_state = self.session_state_cache[session_id]
                        logger.info(f"✓ Using cached state {cached_state} ({self.STATE_NAMES.get(cached_state, '')}) for session {session_id}")
                        state_code = cached_state
                    else:
                        # Look in history for state mentions
                        logger.info("No cached state found, looking in history...")
                        history_state = self._get_state_from_history(
                            user_location=user_location,
                            conversation_history=conversation_history
                        )
                        
                        if history_state:
                            state_code = history_state
                            logger.info(f"✓ Found state {state_code} ({self.STATE_NAMES.get(state_code, '')}) in conversation history")
                            
                            # Cache this state
                            if session_id:
                                self.session_state_cache[session_id] = state_code
                                logger.info(f"✓ Updated session cache from history: {session_id} -> {state_code}")
                        else:
                            logger.info("✗ No state found in conversation history")
                else:
                    # Regular query, no explicit state mention
                    logger.info("Regular query without 'my state' mention")
                    
                    # First try cache if available
                    if session_id and session_id in self.session_state_cache:
                        cached_state = self.session_state_cache[session_id]
                        logger.info(f"✓ Using cached state {cached_state} ({self.STATE_NAMES.get(cached_state, '')}) for session {session_id}")
                        state_code = cached_state
                    else:
                        # Try user location and conversation history
                        logger.info("No cached state, checking user location and history...")
                        state_code = self._get_state_from_history(
                            user_location=user_location,
                            conversation_history=conversation_history
                        )
                        
                        # If we found a state and have a session ID, cache it for future reference
                        if state_code and session_id:
                            self.session_state_cache[session_id] = state_code
                            logger.info(f"✓ Cached state {state_code} for session {session_id}")
                        elif not state_code:
                            logger.info("✗ No state found in user location or history")
            
                # If no state provided but asking about "my state", always ask for the state
                if asking_about_my_state and not state_code:
                    logger.info("User asked about 'my state' but no state could be determined, requesting clarification")
                    return {
                        "text": "To provide information about abortion policies in your state, I need to know which state you're in. Could you please tell me your state name or abbreviation (like 'California' or 'CA'), or provide your ZIP code?",
                        "aspect_type": "policy",
                        "confidence": 0.9,
                        "citations": []
                    }
            
            # If no state provided, return a request for location
            if not state_code:
                logger.info("No state could be determined, requesting state information from user")
                return {
                    "text": "To provide accurate information about abortion policies, I need to know which state you're in. Abortion laws vary significantly by state. Could you please tell me:\n\n1. Your state name or abbreviation (like 'Texas' or 'TX'), or\n2. Your ZIP code\n\nThis will help me give you the most accurate information for your location.",
                    "aspect_type": "policy",
                    "confidence": 0.9,
                    "citations": []
                }
            
            logger.info(f"Processing policy query for state: {state_code}")
            
            # Check cache first
            cache_key = f"{query}_{state_code}"
            if cache_key in self.policy_cache:
                cache_entry = self.policy_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    logger.info(f"Using cached policy response for state {state_code}")
                    response = cache_entry['response']
                    # Add note about multiple states if present
                    if note_about_multiple and "text" in response:
                        response["text"] = response["text"] + "\n\n" + note_about_multiple
                    return response
            
            # Fetch policy data
            policy_data = await self._fetch_policy_data(state_code)
            
            # Generate response
            if self.openai_client:
                response_text = await self._generate_with_openai(query, state_code, policy_data)
            else:
                response_text = self._format_policy_data(state_code, policy_data)
            
            # Add note about multiple states if present
            if note_about_multiple:
                response_text = response_text + "\n\n" + note_about_multiple
            
            # Extract citations
            citations = self._get_policy_citations(state_code)
            
            # Measure processing time
            processing_time = time.time() - start_time
            logger.info(f"Policy query processed in {processing_time:.2f}s")
            
            # Prepare response object
            response = {
                "text": response_text,
                "aspect_type": "policy",
                "state_code": state_code,
                "confidence": 0.9,
                "citations": citations,
                "processing_time": processing_time
            }
            
            # Cache the response
            self.policy_cache[cache_key] = {
                "response": response,
                "timestamp": time.time()
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing policy query: {str(e)}", exc_info=True)
            return {
                "text": "I'm sorry, I'm having trouble accessing the latest policy information. Abortion laws vary by state and may change. You might consider contacting Planned Parenthood for the most current details.",
                "aspect_type": "policy",
                "confidence": 0.5,
                "citations": [{
                    "source": "Planned Parenthood",
                    "url": "https://www.plannedparenthood.org",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }]
            }
    
    def _get_all_state_mentions(self, query: str, full_message: str) -> List[str]:
        """
        Get all state mentions in a query (for handling multi-state queries)
        
        Args:
            query (str): The query text
            full_message (str): The full message text
            
        Returns:
            List[str]: List of state codes found
        """
        message_text = full_message.lower() if full_message else query.lower()
        state_mentions = []
        
        # First look for full state names
        for state_code, state_name in self.STATE_NAMES.items():
            state_pattern = r'\b' + re.escape(state_name.lower()) + r'\b'
            if re.search(state_pattern, message_text):
                state_mentions.append(state_code)
                logger.info(f"Found state name mention: {state_name}")
        
        # Then look for state codes
        for state_code in self.STATE_NAMES.keys():
            # Skip "in" as it's commonly used as a preposition
            if state_code.lower() == "in":
                continue
                
            code_pattern = r'\b' + re.escape(state_code.lower()) + r'\b'
            if re.search(code_pattern, message_text) and state_code not in state_mentions:
                # Double check problematic state codes that could match common words
                if state_code.lower() in ["or", "me", "hi", "ok", "de", "pa", "oh", "id", "co", "wa", "md", "va"]:
                    # Get context around the match to check if it's truly a state reference
                    match = re.search(rf'(\w+\s+)?\b{re.escape(state_code.lower())}\b(\s+\w+)?', message_text)
                    if match:
                        context = match.group(0)
                        if any(term in context for term in ["state", "states", "in", "laws", "abortion"]):
                            state_mentions.append(state_code)
                            logger.info(f"Found state code mention with context: {state_code} in '{context}'")
                else:
                    state_mentions.append(state_code)
                    logger.info(f"Found state code mention: {state_code}")
        
        # Look for ZIP codes
        zip_matches = re.findall(r'\b(\d{5})\b', message_text)
        for zip_code in zip_matches:
            state_from_zip = self._get_state_from_zip(zip_code)
            if state_from_zip and state_from_zip not in state_mentions:
                state_mentions.append(state_from_zip)
                logger.info(f"Found state from ZIP code {zip_code}: {state_from_zip}")
        
        # Remove duplicates while preserving order
        unique_states = []
        for state in state_mentions:
            if state not in unique_states:
                unique_states.append(state)
        
        return unique_states
    
    async def _handle_state_comparison(self, query: str, state_codes: List[str], full_message: str = None) -> Dict[str, Any]:
        """
        Handle comparison between multiple states
        
        Args:
            query (str): The query text
            state_codes (List[str]): List of state codes to compare
            full_message (str): The full message text
            
        Returns:
            Dict[str, Any]: Response with policy comparison
        """
        start_time = time.time()
        
        try:
            # Limit to max 3 states to avoid overload
            if len(state_codes) > 3:
                logger.info(f"Limiting comparison from {len(state_codes)} states to first 3")
                state_codes = state_codes[:3]
            
            # Fetch policy data for all states
            policy_data_dict = {}
            for state_code in state_codes:
                policy_data = await self._fetch_policy_data(state_code)
                policy_data_dict[state_code] = policy_data
            
            # Prepare state names for better readability
            state_names = [f"{self.STATE_NAMES.get(code, code)} ({code})" for code in state_codes]
            
            # Generate comparison text
            if self.openai_client:
                comparison_prompt = f"Compare abortion laws and policies in {', '.join(state_names)}. Highlight key differences in legality, gestational limits, and requirements.\n\n"
                
                # Add policy data for each state
                for state_code, policy_data in policy_data_dict.items():
                    comparison_prompt += f"\n{self.STATE_NAMES.get(state_code, state_code)} ({state_code}) policy data:\n"
                    for key, value in policy_data.items():
                        if key != "citations":
                            comparison_prompt += f"- {key}: {value}\n"
                
                # Generate the comparison using OpenAI
                messages = [
                    {"role": "system", "content": "You are a helpful assistant providing accurate information about abortion policies and laws in the United States. Present information in a clear, factual manner. Focus on key differences between states."},
                    {"role": "user", "content": comparison_prompt}
                ]
                
                response = await self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=1000
                )
                
                comparison_text = response.choices[0].message.content.strip()
            else:
                # Fallback without OpenAI
                comparison_text = f"Comparison of abortion laws in {', '.join(state_names)}:\n\n"
                
                # Simple manual comparison of key factors
                factors = ["legal_status", "gestational_limits", "required_counseling", "waiting_period", "parental_consent_required"]
                
                for factor in factors:
                    comparison_text += f"\n{factor.replace('_', ' ').title()}:\n"
                    for state_code in state_codes:
                        state_name = self.STATE_NAMES.get(state_code, state_code)
                        value = policy_data_dict[state_code].get(factor, "Information not available")
                        comparison_text += f"- {state_name}: {value}\n"
            
            # Combine citations from all states
            all_citations = []
            for state_code in state_codes:
                citations = self._get_policy_citations(state_code)
                for citation in citations:
                    if citation not in all_citations:
                        all_citations.append(citation)
            
            # Measure processing time
            processing_time = time.time() - start_time
            logger.info(f"Policy comparison processed in {processing_time:.2f}s")
            
            # Prepare response object
            response = {
                "text": comparison_text,
                "aspect_type": "policy",
                "state_codes": state_codes,
                "confidence": 0.9,
                "citations": all_citations,
                "processing_time": processing_time
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing policy comparison: {str(e)}", exc_info=True)
            return {
                "text": f"I'm sorry, I'm having trouble comparing abortion policies between {', '.join([self.STATE_NAMES.get(code, code) for code in state_codes])}. Abortion laws vary by state and may change. You might consider contacting Planned Parenthood for the most current details.",
                "aspect_type": "policy",
                "confidence": 0.5,
                "citations": [{
                    "source": "Planned Parenthood",
                    "url": "https://www.plannedparenthood.org",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }]
            }
    
    def _get_state_from_text(self, query: str, full_message: str) -> Optional[str]:
        """
        Extract state code from the current query text
        
        Args:
            query (str): The specific policy query aspect
            full_message (str): The original complete user message
            
        Returns:
            Optional[str]: State code or None
        """
        query_lower = full_message.lower() if full_message else query.lower()
        
        # Check if the message consists solely of a state name or code (likely a follow-up query)
        query_stripped = query_lower.strip().rstrip('.!?')
        
        # First priority: standalone state names like "California" or "New York"
        words = query_stripped.split()
        if len(words) <= 3:  # State name should be 1-3 words
            for state_code, state_name in self.STATE_NAMES.items():
                if state_name.lower() == query_stripped:
                    logger.info(f"Found exact standalone state name: {state_name}")
                    return state_code
        
        # Second priority: standalone state code like "CA" or "NY"
        query_upper = query_stripped.upper()
        if len(query_stripped) == 2 and query_upper in self.STATE_NAMES:
            logger.info(f"Found exact standalone state code: {query_upper}")
            return query_upper
        
        # Skip phrases like "in my state", "in your state" which could falsely match with "in"
        if re.search(r'\bin\s+(?:my|your|their|our)\s+state\b', query_lower):
            logger.info("Detected 'in my/your/their state' phrase, skipping state code search in this area")
            # Remove this pattern before proceeding with other checks
            query_lower = re.sub(r'\bin\s+(?:my|your|their|our)\s+state\b', '', query_lower)
        
        # Third priority: state name mentioned within a query
        for state_code, state_name in self.STATE_NAMES.items():
            state_pattern = r'\b' + re.escape(state_name.lower()) + r'\b'
            if re.search(state_pattern, query_lower):
                logger.info(f"Found state name in query: {state_name}")
                return state_code
        
        # Fourth priority: state code mentioned within a query (with special care for small codes)
        for state_code in self.STATE_NAMES.keys():
            # Skip "in" as a standalone word since it's commonly used in phrases
            if state_code.lower() == "in" and re.search(r'\bin\b', query_lower):
                continue
                
            # Use word boundaries to ensure we match standalone state codes
            code_pattern = r'\b' + re.escape(state_code.lower()) + r'\b'
            if re.search(code_pattern, query_lower):
                # Double check problematic state codes that could match common words
                if state_code.lower() in ["in", "or", "me", "hi", "ok", "de", "pa", "oh", "id", "co", "wa", "md", "va"]:
                    # Get context around the match to check if it's truly a state reference
                    match = re.search(rf'(\w+\s+)?\b{re.escape(state_code.lower())}\b(\s+\w+)?', query_lower)
                    if match:
                        context = match.group(0)
                        # Skip prepositions like "in the", "in my", etc.
                        if state_code.lower() == "in" and re.search(r'\bin\s+(the|my|your|a|an|this|that)\b', context):
                            logger.info(f"Skipping state code match '{state_code}' as it looks like a preposition: '{context}'")
                            continue
                
                logger.info(f"Found state code in query: {state_code}")
                return state_code
        
        # Fifth priority: check for ZIP codes
        state_from_zip = self._get_state_from_zip(query=full_message if full_message else query)
        if state_from_zip:
            logger.info(f"Found state from ZIP code: {state_from_zip}")
            return state_from_zip
        
        # Look for "in [state]" pattern but be careful to avoid "in my state" phrases
        state_in_query = re.search(r'in\s+([a-zA-Z\s]+?)(?:\s|$|\.|\?|,)', query_lower)
        if state_in_query:
            potential_state = state_in_query.group(1).strip()
            
            # Skip phrases like "my state", "your state", etc.
            if re.search(r'\b(my|your|our|their)\s+state\b', potential_state):
                logger.info(f"Skipping 'in [my/your/their] state' phrase: '{potential_state}'")
                return None
            
            # Check if it's a state code
            potential_state_upper = potential_state.upper()
            if len(potential_state) == 2 and potential_state_upper in self.STATE_NAMES:
                logger.info(f"Found exact state code in 'in STATE' pattern: {potential_state_upper}")
                return potential_state_upper
                
            # Check if it's a state name
            for state_code, state_name in self.STATE_NAMES.items():
                if state_name.lower() == potential_state:
                    logger.info(f"Found exact state match in 'in STATE' pattern: {state_name}")
                    return state_code
            
            # Try partial matches for state names
            best_match = None
            highest_similarity = 0
            
            for state_code, state_name in self.STATE_NAMES.items():
                state_name_lower = state_name.lower()
                if state_name_lower in potential_state or potential_state in state_name_lower:
                    similarity = len(potential_state) / max(len(potential_state), len(state_name_lower))
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = state_code
            
            if best_match and highest_similarity > 0.7:  # Higher threshold to avoid false matches
                logger.info(f"Found partial state match: {self.STATE_NAMES[best_match]} ({highest_similarity:.2f} similarity)")
                return best_match
        
        # No state found in text
        return None
    
    def _get_state_from_history(self, user_location: Optional[Dict[str, str]], 
                               conversation_history: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        """
        Extract state code from user location or conversation history
        
        Args:
            user_location (Optional[Dict[str, str]]): User's location data
            conversation_history (Optional[List[Dict[str, Any]]]): Conversation history
            
        Returns:
            Optional[str]: State code or None
        """
        # 1. Try user location first if provided
        if user_location:
            # Check for state code
            state_code = user_location.get('state', '').upper()
            if state_code and len(state_code) == 2 and state_code in self.STATE_NAMES:
                logger.info(f"Found state code in user location: {state_code}")
                return state_code
                
            # Check for state name
            state_name = user_location.get('state', '')
            for code, name in self.STATE_NAMES.items():
                if name.lower() == state_name.lower():
                    logger.info(f"Found state name in user location: {name}")
                    return code
            
            # Check for ZIP code in user location
            zip_code = user_location.get('zip', '') or user_location.get('postal_code', '')
            if zip_code and len(zip_code) >= 5:
                state_from_zip = self._get_state_from_zip(query=zip_code)
                if state_from_zip:
                    logger.info(f"Found state from ZIP code in user location: {state_from_zip}")
                    return state_from_zip
        
        # 2. Check conversation history
        if conversation_history:
            # Look at the last 5 messages from the user
            user_msgs = [msg.get('message', '').lower() 
                         for msg in conversation_history[-5:] 
                         if msg.get('sender') == 'user']
            
            for msg in user_msgs:
                # Try each message for state mentions
                state_found = self._get_state_from_text(query=msg, full_message=msg)
                if state_found:
                    logger.info(f"Found state {state_found} in conversation history")
                    return state_found
                
                # Check for standalone state names or abbreviations in short messages
                if len(msg.split()) <= 3:  # Short message like "Texas" or "TX"
                    msg_cleaned = msg.strip().rstrip('.!?')
                    msg_upper = msg_cleaned.upper()
                    
                    # Check for exact state code
                    if len(msg_cleaned) == 2 and msg_upper in self.STATE_NAMES:
                        logger.info(f"Found standalone state code in history: {msg_upper}")
                        return msg_upper
                    
                    # Check for exact state name
                    for state_code, state_name in self.STATE_NAMES.items():
                        if state_name.lower() == msg_cleaned.lower():
                            logger.info(f"Found standalone state name in history: {state_name}")
                            return state_code
        
        # No state found
        return None
    
    async def _fetch_policy_data(self, state_code: str) -> Dict[str, Any]:
        """
        Fetch policy data from the Abortion Policy API
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            Dict[str, Any]: Policy data
        """
        try:
            headers = {"token": self.abortion_policy_api_key}
            policy_info = {"endpoints": {}}
            
            # Ensure the API base URL doesn't end with a slash for consistent URL formatting
            api_base = self.api_base_url.rstrip('/')
            
            # Use aiohttp for asynchronous API calls
            async with aiohttp.ClientSession() as session:
                # Make multiple API calls in parallel
                tasks = []
                for key, endpoint in self.api_endpoints.items():
                    url = f"{api_base}/{endpoint}/states/{state_code}"
                    tasks.append(self._fetch_endpoint(session, url, headers, key))
                
                # Wait for all API calls to complete
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Process results
                for key, result in zip(self.api_endpoints.keys(), results):
                    if isinstance(result, Exception):
                        logger.error(f"Error fetching {key} for {state_code}: {str(result)}")
                        policy_info["endpoints"][key] = {}
                    else:
                        policy_info["endpoints"][key] = result
            
            # If we have data, add state information
            if any(policy_info["endpoints"].values()):
                policy_info["state_code"] = state_code
                policy_info["state_name"] = self.STATE_NAMES.get(state_code, state_code)
                logger.info(f"Successfully fetched policy data for {state_code}")
                return policy_info
            else:
                logger.error(f"No policy data found for {state_code}")
                return self._get_fallback_policy_data(state_code)
                
        except Exception as e:
            logger.error(f"Error fetching policy data for {state_code}: {str(e)}", exc_info=True)
            return self._get_fallback_policy_data(state_code)
    
    async def _fetch_endpoint(self, session, url, headers, key):
        """
        Fetch data from a specific endpoint
        
        Args:
            session: aiohttp ClientSession
            url: API endpoint URL
            headers: Request headers
            key: Endpoint key
            
        Returns:
            dict: API response data
        """
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"API returned status {response.status} for {url}")
                    return {}
        except Exception as e:
            logger.error(f"Error fetching endpoint {key}: {str(e)}")
            return {}
    
    def _get_fallback_policy_data(self, state_code: str) -> Dict[str, Any]:
        """
        Get fallback policy data when API fails
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            Dict[str, Any]: Fallback policy data
        """
        state_name = self.STATE_NAMES.get(state_code, state_code)
        return {
            "state_code": state_code,
            "state_name": state_name,
            "error": True,
            "endpoints": {},
            "resources": [
                "Planned Parenthood",
                "National Abortion Federation",
                "INeedAnA.com"
            ],
            "sources": [
                {
                    "title": "Planned Parenthood",
                    "url": "https://www.plannedparenthood.org/",
                    "accessed_date": datetime.now().strftime('%Y-%m-%d')
                }
            ]
        }
    
    async def _generate_with_openai(self, query: str, state_code: str, 
                                   policy_data: Dict[str, Any]) -> str:
        """
        Generate a policy response using OpenAI
        
        Args:
            query (str): The user's query
            state_code (str): Two-letter state code
            policy_data (Dict[str, Any]): Policy data
            
        Returns:
            str: Generated response
        """
        if not self.openai_client:
            raise ValueError("OpenAI client not available")
        
        # Convert policy data to a string
        policy_json = json.dumps(policy_data, indent=2)
        state_name = self.STATE_NAMES.get(state_code, state_code)
        
        system_message = f"""You are an expert reproductive health assistant specialized in abortion policy information.
Use the provided JSON data to answer questions about abortion policy in {state_name}.
Be accurate, factual, and concise. Specify any limitations or restrictions clearly.
If the data doesn't contain certain information, acknowledge that gap.
IMPORTANT: Focus ONLY on the provided data for {state_name}. Do not reference policies for other states.
Format your response in a clear, easy-to-understand way for someone seeking policy information."""
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o",  # Use the most advanced model for best quality
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": f"Policy data for {state_name}:\n{policy_json}\n\nQuestion: {query}"}
                ],
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=600
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Error generating policy response with OpenAI: {str(e)}", exc_info=True)
            # Fall back to template response
            return self._format_policy_data(state_code, policy_data)
    
    def _format_policy_data(self, state_code: str, policy_data: Dict[str, Any]) -> str:
        """
        Format policy data into a readable response
        
        Args:
            state_code (str): Two-letter state code
            policy_data (Dict[str, Any]): Policy data
            
        Returns:
            str: Formatted response
        """
        state_name = self.STATE_NAMES.get(state_code, state_code)
        
        # Check if we have an error
        if policy_data.get("error", False):
            return (
                f"I'm sorry, I'm having trouble accessing the latest policy information for {state_name}. "
                f"Abortion laws vary by state and may change frequently. "
                f"You might consider contacting Planned Parenthood or check abortionfinder.org for the most current details."
            )
        
        # Start building the response
        response = f"Here's the abortion policy information for {state_name}:\n\n"
        
        # Add gestational limits
        gestational_data = policy_data.get("endpoints", {}).get("gestational_limits", {})
        if gestational_data:
            if gestational_data.get("banned", False):
                response += f"Abortion is banned in {state_name}."
            elif "banned_after_weeks_since_LMP" in gestational_data:
                weeks = gestational_data["banned_after_weeks_since_LMP"]
                response += f"In {state_name}, abortion is prohibited after {weeks} weeks since last menstrual period."
            else:
                response += f"I don't have specific information about gestational limits in {state_name}."
        
        # Add waiting periods
        waiting_data = policy_data.get("endpoints", {}).get("waiting_periods", {})
        if waiting_data and "waiting_period_hours" in waiting_data:
            hours = waiting_data["waiting_period_hours"]
            if hours > 0:
                response += f" There is a required waiting period of {hours} hours."
        
        # Add insurance coverage information
        insurance_data = policy_data.get("endpoints", {}).get("insurance_coverage", {})
        if insurance_data:
            if insurance_data.get("private_coverage_prohibited", False):
                response += " Private insurance coverage for abortion is prohibited."
            if insurance_data.get("exchange_coverage_prohibited", False):
                response += " Insurance purchased through the health exchange cannot cover abortion."
            if insurance_data.get("medicaid_coverage_provider", "") == "yes":
                response += " Medicaid does provide coverage for abortion in certain circumstances."
        
        # Add minors restrictions
        minors_data = policy_data.get("endpoints", {}).get("minors", {})
        if minors_data:
            if minors_data.get("parental_consent_required", False):
                response += " Parental consent is required for minors seeking abortion."
            if minors_data.get("parental_notification_required", False):
                response += " Parents must be notified before a minor can obtain an abortion."
        
        # Add resources and disclaimer
        response += (
            "\n\nThis information is based on the most recent data available, but laws may have changed. "
            "For the most up-to-date information, please contact Planned Parenthood or visit abortionfinder.org."
        )
        
        return response
    
    def _get_policy_citations(self, state_code: str) -> List[Dict[str, Any]]:
        """
        Get citations for policy data
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            List[Dict[str, Any]]: Citations
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        return [
            {
                "source": "Abortion Policy API",
                "url": "https://www.abortionpolicyapi.com/",
                "accessed_date": today
            },
            {
                "source": "Planned Parenthood",
                "url": "https://www.plannedparenthood.org/learn/abortion/abortion-laws-by-state",
                "accessed_date": today
            },
            {
                "source": "Abortion Finder",
                "url": "https://www.abortionfinder.org/",
                "accessed_date": today
            }
        ]
    
    def _get_state_from_zip(self, query: str) -> Optional[str]:
        """
        Extract state code from a ZIP code in the query using zipcodes library
        
        Args:
            query (str): The query text that might contain a ZIP code
            
        Returns:
            Optional[str]: State code or None
        """
        try:
            # Import Preprocessor here to avoid circular dependencies
            from .preprocessor import Preprocessor
            
            # Check if we have initialized a preprocessor
            if not hasattr(self, '_preprocessor'):
                self._preprocessor = Preprocessor()
                logger.info("Initialized Preprocessor for ZIP code lookup")
            
            # Look for 5-digit ZIP code pattern
            zip_matches = re.findall(r'\b(\d{5})\b', query)
            
            if not zip_matches:
                return None
            
            # Try each ZIP code found in the text using the preprocessor
            for zip_code in zip_matches:
                state = self._preprocessor.get_state_from_zip(zip_code)
                if state:
                    logger.info(f"Matched ZIP code {zip_code} to state {state} using preprocessor")
                    return state
                        
            # No valid ZIP code found, fall back to our original implementation
            return self._get_state_from_zip_fallback(query)
            
        except ImportError:
            logger.error("Could not import Preprocessor, falling back to simplified approach")
            return self._get_state_from_zip_fallback(query)
    
    def _get_state_from_zip_fallback(self, query: str) -> Optional[str]:
        """
        Fallback method using a simplified mapping of ZIP codes to states
        
        Args:
            query (str): The query text that might contain a ZIP code
            
        Returns:
            Optional[str]: State code or None
        """
        # Look for 5-digit ZIP code pattern
        zip_matches = re.findall(r'\b(\d{5})\b', query)
        
        if not zip_matches:
            return None
            
        # Simple mapping of ZIP code ranges to states
        zip_ranges = {
            'AL': (35000, 36999),
            'AK': (99500, 99999),
            'AZ': (85000, 86999),
            'AR': (71600, 72999),
            'CA': (90000, 96699),
            'CO': (80000, 81999),
            'CT': (6000, 6999),
            'DE': (19700, 19999),
            'DC': (20000, 20599),
            'FL': (32000, 34999),
            'GA': (30000, 31999),
            'HI': (96700, 96899),
            'ID': (83200, 83999),
            'IL': (60000, 62999),
            'IN': (46000, 47999),
            'IA': (50000, 52999),
            'KS': (66000, 67999),
            'KY': (40000, 42799),
            'LA': (70000, 71599),
            'ME': (3900, 4999),
            'MD': (20600, 21999),
            'MA': (1000, 2799),
            'MI': (48000, 49999),
            'MN': (55000, 56999),
            'MS': (38600, 39999),
            'MO': (63000, 65999),
            'MT': (59000, 59999),
            'NE': (68000, 69999),
            'NV': (89000, 89999),
            'NH': (3000, 3899),
            'NJ': (7000, 8999),
            'NM': (87000, 88499),
            'NY': (10000, 14999),
            'NC': (27000, 28999),
            'ND': (58000, 58999),
            'OH': (43000, 45999),
            'OK': (73000, 74999),
            'OR': (97000, 97999),
            'PA': (15000, 19699),
            'RI': (2800, 2999),
            'SC': (29000, 29999),
            'SD': (57000, 57999),
            'TN': (37000, 38599),
            'TX': (75000, 79999),
            'UT': (84000, 84999),
            'VT': (5000, 5999),
            'VA': (22000, 24699),
            'WA': (98000, 99499),
            'WV': (24700, 26999),
            'WI': (53000, 54999),
            'WY': (82000, 83199)
        }
        
        for zip_code in zip_matches:
            zip_int = int(zip_code)
            
            for state, (lower, upper) in zip_ranges.items():
                if lower <= zip_int <= upper:
                    logger.info(f"Matched ZIP code {zip_code} to state {state} using fallback")
                    return state
        
        return None 