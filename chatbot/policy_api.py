import os
import logging
import requests
import json
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class PolicyAPI:
    """
    Integration with the abortion policy API to provide up-to-date policy information
    """
    def __init__(self):
        """Initialize the Policy API client"""
        logger.info("Initializing Policy API")
        self.api_key = os.environ.get("ABORTION_POLICY_API_KEY")
        self.base_url = "https://api.abortionpolicyapi.com/v1"
        
        if not self.api_key:
            logger.warning("Abortion Policy API key not found in environment variables")
        
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
            logger.error(f"Error extracting state: {str(e)}")
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
            return {"error": "API key not configured"}
        
        try:
            headers = {
                "x-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            
            url = f"{self.base_url}/policy_by_state/{state_code.upper()}"
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return {"error": f"API returned status code {response.status_code}"}
                
        except Exception as e:
            logger.error(f"Error fetching policy data: {str(e)}")
            return {"error": str(e)}
    
    def get_policy_response(self, question: str) -> str:
        """
        Process a policy-related question and return a response
        
        Args:
            question (str): User's question about abortion policy
            
        Returns:
            str: Formatted response with policy information
        """
        # Extract state from question
        state_code = self._extract_state_from_question(question)
        
        if not state_code:
            return self._get_general_policy_information(question)
        
        # Get policy data for the state
        policy_data = self.get_policy_data(state_code)
        
        if "error" in policy_data:
            logger.warning(f"Error in policy data: {policy_data['error']}")
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
        
        # Convert policy data to readable format
        formatted_data = json.dumps(policy_data, indent=2)
        
        prompt = f"""
        The user asked: "{question}"
        
        This question is about abortion policy in the state with code {state_code}.
        
        Here is the policy data for this state:
        {formatted_data}
        
        Please provide a clear, accurate response that directly answers the user's question using this policy data.
        
        Format the response in a user-friendly way with appropriate headings and bullet points.
        Be sure to mention that this information is current according to the API data, but policies may change.
        Include a disclaimer that this is for informational purposes only and not legal advice.
        
        If the policy data doesn't specifically address the user's question, acknowledge that and provide the most relevant information available.
        """
        
        try:
            return self.gpt_model.get_response(prompt)
        except Exception as e:
            logger.error(f"Error formatting policy response: {str(e)}")
            
            # Fallback response using the raw data
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
            
            response = f"Here is information about abortion policy in {state_name}:\n\n"
            
            # Extract key information from policy data
            if "banned" in policy_data:
                response += f"• Abortion banned: {policy_data['banned']}\n"
                
            if "gestational_limit_in_weeks" in policy_data:
                limit = policy_data['gestational_limit_in_weeks']
                if limit:
                    response += f"• Gestational limit: {limit} weeks\n"
                else:
                    response += "• No specific gestational limit mentioned in the data\n"
                    
            if "waiting_period_in_hours" in policy_data:
                waiting = policy_data['waiting_period_in_hours']
                if waiting:
                    response += f"• Waiting period required: {waiting} hours\n"
                    
            if "counseling" in policy_data:
                response += f"• Counseling required: {policy_data['counseling']}\n"
                
            response += "\nThis information is provided for informational purposes only and is not legal advice. Policies may change, so please consult with a healthcare provider or legal professional for the most current information."
            
            return response