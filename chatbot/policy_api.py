import os
import logging
import requests
import json
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)

class PolicyAPI:
    """
    Integration with the abortion policy API to provide up-to-date policy information
    """
    # State code to full name mapping
    STATE_NAMES = {
        "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
        "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
        "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
        "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
        "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
        "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
        "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
        "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
        "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
        "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
        "DC": "District of Columbia"
    }
    
    def __init__(self):
        """Initialize the Policy API client"""
        logger.info("Initializing Policy API")
        # Check for various environment variable names for the API key
        self.api_key = os.environ.get("ABORTION_POLICY_API_KEY") or os.environ.get("POLICY_API_KEY")
        self.base_url = "https://api.abortionpolicyapi.com/v1"
        
        if not self.api_key:
            logger.warning("Abortion Policy API key not found in environment variables")
        else:
            logger.info("Abortion Policy API key found")
        
        self.gpt_model = None
        logger.info("Policy API initialized successfully")
        
    def _extract_state_from_question(self, question: str) -> Optional[str]:
        """
        Extract state information from a user's question
        
        Args:
            question (str): User's question about abortion policy
            
        Returns:
            Optional[str]: State code if found, None otherwise
        """
        # First try a rule-based approach for common patterns
        state_patterns = {
            "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", 
            "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE", 
            "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID", 
            "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS", 
            "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", 
            "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS", 
            "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV", 
            "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY", 
            "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK", 
            "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC", 
            "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", 
            "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV", 
            "wisconsin": "WI", "wyoming": "WY", "district of columbia": "DC", 
            "washington dc": "DC", "dc": "DC"
        }
        
        # Also check for state abbreviations (in isolation)
        abbrev_pattern = r'\b([A-Za-z]{2})\b'
        import re
        
        question_lower = question.lower()
        
        # Check for state names
        for state_name, code in state_patterns.items():
            if state_name in question_lower:
                logger.debug(f"Found state name {state_name} in question")
                return code
        
        # Check for state abbreviations (case insensitive)
        abbrevs = re.findall(abbrev_pattern, question)
        if abbrevs:
            # Convert to uppercase for consistency
            potential_abbrevs = [abbr.upper() for abbr in abbrevs]
            logger.debug(f"Found potential state abbreviations: {potential_abbrevs}")
            
            # Check if any of the found abbreviations are valid state codes
            valid_state_codes = set(state_patterns.values())
            for abbr in potential_abbrevs:
                if abbr in valid_state_codes:
                    logger.debug(f"Matched valid state code: {abbr}")
                    return abbr
        
        # Fall back to GPT for more complex cases
        from chatbot.gpt_integration import GPTModel
        
        if not self.gpt_model:
            self.gpt_model = GPTModel()
        
        prompt = f"""
        Extract the US state mentioned in this question. Return only the state's two-letter code in uppercase.
        If no specific state is mentioned, return "NONE".
        
        Question: {question}
        
        State code (e.g., CA, TX, NY, or NONE):
        """
        
        try:
            response = self.gpt_model.get_response(prompt).strip()
            if response == "NONE":
                return None
            return response if len(response) == 2 else None
        except Exception as e:
            logger.error(f"Error extracting state from GPT: {str(e)}")
            return None
    
    def get_policy_data(self, state_code: str) -> Dict[str, Any]:
        """
        Get abortion policy data for a specific state
        
        Args:
            state_code (str): Two-letter state code
            
        Returns:
            Dict: Policy data for the state
        """
        if not self.api_key:
            logger.error("API key not found when trying to fetch policy data")
            return {"error": "API key not configured"}
        
        try:
            # Define endpoints to fetch
            endpoints = {
                "waiting_periods": "waiting_periods",
                "insurance_coverage": "insurance_coverage",
                "gestational_limits": "gestational_limits",
                "minors": "minors"
            }
            
            # Ensure state code is uppercase
            state_code = state_code.upper()
            
            # Use 'token' in headers as discovered in testing
            headers = {"token": self.api_key}
            
            # Log the API key (first few characters only for security)
            masked_key = self.api_key[:3] + "*" * (len(self.api_key) - 3) if self.api_key else "None"
            logger.debug(f"Using API key: {masked_key}")
            
            # Combined policy data object
            policy_data = {
                "state_code": state_code,
                "endpoints": {}
            }
            
            # Collect data from all endpoints
            for key, endpoint in endpoints.items():
                url = f"{self.base_url}/{endpoint}/states/{state_code}"
                logger.debug(f"Making request to endpoint: {url}")
                
                # Add slight delay to avoid rate limiting
                import time
                time.sleep(0.5)
                
                try:
                    response = requests.get(url, headers=headers)
                    logger.debug(f"API Response status for {key}: {response.status_code}")
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data:  # if data is not empty
                            policy_data["endpoints"][key] = data
                            # Set success flag since we got data from at least one endpoint
                            policy_data["success"] = True
                    else:
                        logger.warning(f"Endpoint {key} failed with status {response.status_code}: {response.text}")
                        policy_data["endpoints"][key] = {"error": f"Status code {response.status_code}"}
                except Exception as endpoint_error:
                    logger.error(f"Error fetching {key} endpoint: {str(endpoint_error)}")
                    policy_data["endpoints"][key] = {"error": str(endpoint_error)}
            
            # Check if we got any data at all
            if not policy_data.get("success") and not any(endpoint for endpoint in policy_data["endpoints"].values() if "error" not in endpoint):
                logger.error("No policy data retrieved from any endpoint")
                return {"error": "Failed to retrieve policy data from any endpoint", "state_attempted": state_code}
                
            return policy_data
                
        except Exception as e:
            logger.error(f"Error fetching policy data: {str(e)}", exc_info=True)
            return {"error": str(e)}
    
    def get_policy_response(self, question: str, conversation_history: Optional[List[Dict[str, Any]]] = None, location_context: Optional[str] = None) -> str:
        """
        Process a policy-related question and return a response
        
        Args:
            question (str): User's question about abortion policy
            conversation_history (list, optional): List of previous messages in the conversation
            location_context (str, optional): Explicit location context if already detected
            
        Returns:
            str: Formatted response with policy information
        """
        # First use the location_context if provided
        state_code = None
        if location_context:
            state_code = location_context
            logger.info(f"Using provided location context: {state_code}")
            
        # If no location context, try to extract state from the current question
        if not state_code:
            state_code = self._extract_state_from_question(question)
            logger.info(f"Checking for state in question: '{question}' - Result: {state_code}")
        
        # If still no state, try to find it in conversation history
        if not state_code and conversation_history:
            logger.info("No state found in current question, checking conversation history")
            
            # Use the full state dictionary instead of a limited set
            all_states = state_patterns.copy()
            
            # Iterate through conversation history from newest to oldest
            for message in reversed(conversation_history):
                if message['sender'] == 'user':
                    msg = message['message'].lower()
                    logger.info(f"Checking history message: '{msg}'")
                    
                    # First try direct match with all states
                    for state_name, code in all_states.items():
                        if state_name in msg:
                            logger.info(f"Direct match! Found state {state_name} in message")
                            state_code = code
                            break
                    
                    # If we found a state, break out of the messages loop
                    if state_code:
                        break
                        
                    # Otherwise try the full extraction
                    potential_state = self._extract_state_from_question(message['message'])
                    if potential_state:
                        logger.info(f"Found state {potential_state} in conversation history")
                        state_code = potential_state
                        break
        
        if not state_code:
            logger.debug("No state found in question or conversation history")
            return self._get_general_policy_information(question)
        
        logger.info(f"Getting policy data for state: {state_code}")
        # Get policy data for the state
        policy_data = self.get_policy_data(state_code)
        
        if "error" in policy_data:
            logger.warning(f"Error in policy data: {policy_data['error']}")
            if "state_attempted" in policy_data:
                state_name = state_names.get(policy_data["state_attempted"].upper(), policy_data["state_attempted"])
                logger.info(f"Failed to get data for {state_name}, returning state-specific general information")
                # Return a tailored response for the state that doesn't cite the API
                return f"I'm sorry, I'm having trouble accessing specific policy information for {state_name} at the moment. Abortion policies vary by state and may change. For the most accurate and up-to-date information about abortion access in {state_name}, I recommend contacting Planned Parenthood directly or visiting their website. They can provide current information about options available to you in {state_name}."
            return self._get_general_policy_information(question)
        
        # Format the policy data into a user-friendly response
        return self._format_policy_response(question, state_code, policy_data)
    
    def _get_general_policy_information(self, question: str) -> str:
        """
        Provide general policy information when state-specific data is unavailable
        
        Args:
            question (str): User's original question
            
        Returns:
            str: General policy information
        """
        from chatbot.gpt_integration import GPTModel
        
        if not self.gpt_model:
            self.gpt_model = GPTModel()
            
        prompt = f"""
        The user has asked the following policy-related question, but I couldn't identify a specific state or couldn't retrieve state-specific policy data:
        
        "{question}"
        
        Please provide a general response about abortion policies in the United States, explaining that policies vary by state and suggesting that the user specify a state for more detailed information. Include information about how to find state-specific resources.
        
        Keep the response factual, neutral, and informative. Avoid including specific policy details that might be outdated or incorrect.
        """
        
        try:
            return self.gpt_model.get_response(prompt)
        except Exception as e:
            logger.error(f"Error getting general policy info: {str(e)}")
            return "I'm sorry, I'm having trouble providing policy information at the moment. Abortion policies vary significantly by state. For the most accurate and up-to-date information, consider visiting the Planned Parenthood website or contacting a healthcare provider in your state."
    
    def _format_policy_response(self, question: str, state_code: str, policy_data: Dict[str, Any]) -> str:
        """
        Format policy API data into a user-friendly response
        
        Args:
            question (str): User's original question
            state_code (str): Two-letter state code
            policy_data (dict): Policy data from the API
            
        Returns:
            str: Formatted response
        """
        from chatbot.gpt_integration import GPTModel
        
        if not self.gpt_model:
            self.gpt_model = GPTModel()
        
        # Get state name for better readability
        state_names = {
            "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", 
            "CA": "California", "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", 
            "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", 
            "IL": "Illinois", "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", 
            "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland", 
            "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", 
            "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", 
            "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York", 
            "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", 
            "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", 
            "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", 
            "VT": "Vermont", "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", 
            "WI": "Wisconsin", "WY": "Wyoming", "DC": "District of Columbia"
        }
        state_name = state_names.get(state_code.upper(), state_code)
        
        # Convert policy data to readable format with special parsing for our new structure
        formatted_data = {
            "state_name": state_name,
            "state_code": state_code
        }
        
        # Extract data from each endpoint
        endpoints_data = policy_data.get("endpoints", {})
        
        # Extract waiting periods info
        if "waiting_periods" in endpoints_data:
            wp_data = endpoints_data["waiting_periods"].get(state_name, {})
            formatted_data["waiting_periods"] = wp_data
        
        # Extract insurance coverage info
        if "insurance_coverage" in endpoints_data:
            ic_data = endpoints_data["insurance_coverage"].get(state_name, {})
            formatted_data["insurance_coverage"] = ic_data
        
        # Extract gestational limits info
        if "gestational_limits" in endpoints_data:
            gl_data = endpoints_data["gestational_limits"].get(state_name, {})
            formatted_data["gestational_limits"] = gl_data
            
            # Extract key gestational limit info for the prompt
            if "banned_after_weeks_since_LMP" in gl_data:
                formatted_data["weeks_limit"] = gl_data["banned_after_weeks_since_LMP"]
            
            if "exception_life" in gl_data:
                formatted_data["exception_life"] = gl_data["exception_life"]
                
            if "exception_health" in gl_data:
                formatted_data["exception_health"] = gl_data["exception_health"]
        
        # Extract minors info
        if "minors" in endpoints_data:
            minors_data = endpoints_data["minors"].get(state_name, {})
            formatted_data["minors"] = minors_data
            
            # Extract key minors info
            if "allows_minor_to_consent_to_abortion" in minors_data:
                formatted_data["minor_consent"] = minors_data["allows_minor_to_consent_to_abortion"]
        
        # Format the data as JSON string for the prompt
        formatted_json = json.dumps(formatted_data, indent=2)
        
        prompt = f"""
        The user asked: "{question}"
        
        This question is about abortion policy in {state_name} (state code: {state_code}).
        
        Here is the policy data from the Abortion Policy API for this state:
        {formatted_json}
        
        Please provide a clear, accurate response that directly answers the user's question using this policy data.
        Focus on the specific policy areas the question is about.
        
        Format the response in a user-friendly way with appropriate headings and bullet points.
        Include appropriate citations to the Abortion Policy API.
        Be sure to mention that this information is current according to the API data, but policies may change.
        Include a disclaimer that this is for informational purposes only and not legal advice.
        
        If the policy data doesn't specifically address the user's question, acknowledge that and provide the most relevant information available.
        """
        
        try:
            # Add a citation to the abortion policy API
            from chatbot.citation_manager import CitationManager
            citation_mgr = CitationManager()
            response = self.gpt_model.get_response(prompt)
            return citation_mgr.add_citation_to_text(response, "abortion_policy_api")
        except Exception as e:
            logger.error(f"Error formatting policy response: {str(e)}")
            
            # Fallback response using the raw data
            response = f"Here is information about abortion policy in {state_name}:\n\n"
            
            # Format each section of the policy data
            if "gestational_limits" in endpoints_data:
                gl_data = endpoints_data["gestational_limits"].get(state_name, {})
                response += "## Gestational Limits\n"
                
                if "banned_after_weeks_since_LMP" in gl_data:
                    weeks = gl_data["banned_after_weeks_since_LMP"]
                    if weeks == 99:  # API uses 99 to indicate no specific ban
                        response += "• No specific week limit mentioned\n"
                    else:
                        response += f"• Abortion banned after {weeks} weeks since last menstrual period\n"
                        
                if "exception_life" in gl_data:
                    response += f"• Exception for life of the pregnant person: {gl_data['exception_life']}\n"
                    
                if "exception_health" in gl_data:
                    response += f"• Health exception: {gl_data['exception_health']}\n"
                    
                response += "\n"
                
            if "waiting_periods" in endpoints_data:
                wp_data = endpoints_data["waiting_periods"].get(state_name, {})
                if wp_data:
                    response += "## Waiting Periods\n"
                    for key, value in wp_data.items():
                        if key != "Last Updated":
                            response += f"• {key.replace('_', ' ').title()}: {value}\n"
                    response += "\n"
                    
            if "insurance_coverage" in endpoints_data:
                ic_data = endpoints_data["insurance_coverage"].get(state_name, {})
                if ic_data:
                    response += "## Insurance Coverage\n"
                    for key, value in ic_data.items():
                        if key != "Last Updated":
                            response += f"• {key.replace('_', ' ').title()}: {value}\n"
                    response += "\n"
                    
            if "minors" in endpoints_data:
                minors_data = endpoints_data["minors"].get(state_name, {})
                if minors_data:
                    response += "## Minors\n"
                    for key, value in minors_data.items():
                        if key != "Last Updated":
                            response += f"• {key.replace('_', ' ').title()}: {value}\n"
                    response += "\n"
                
            response += "This information is provided for informational purposes only and is not legal advice. Policies may change, so please consult with a healthcare provider or legal professional for the most current information."
            
            # Add a citation to the abortion policy API
            from chatbot.citation_manager import CitationManager
            citation_mgr = CitationManager()
            return citation_mgr.add_citation_to_text(response, "abortion_policy_api")