import os
import logging
import requests
import json

logger = logging.getLogger(__name__)

class PolicyAPI:
    """
    API client for fetching abortion policy information by state
    """
    def __init__(self):
        """Initialize the Policy API client"""
        logger.info("Initializing Policy API")
        try:
            # The API endpoint would typically be set in environment variables
            self.api_endpoint = os.environ.get("POLICY_API_ENDPOINT", "https://api.example.org/abortion-policy")
            self.api_key = os.environ.get("POLICY_API_KEY", "default_key")
            
            logger.info("Policy API initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Policy API: {str(e)}", exc_info=True)
            raise
    
    def get_policy_response(self, question):
        """
        Get policy information based on the user's question
        
        Args:
            question (str): User's policy-related question
        
        Returns:
            str: Response with policy information
        """
        try:
            # Extract state information from the question if present
            state = self._extract_state_from_question(question)
            
            if not state:
                return self._get_general_policy_response(question)
            
            # Make API request for state-specific policy
            return self._get_state_policy(state, question)
            
        except Exception as e:
            logger.error(f"Error getting policy response: {str(e)}", exc_info=True)
            return "I'm sorry, I'm having trouble retrieving the latest policy information. For the most up-to-date and accurate information, please consult the Planned Parenthood website or contact a healthcare provider directly."
    
    def _extract_state_from_question(self, question):
        """
        Extract state name from the question
        
        Args:
            question (str): User's question
        
        Returns:
            str or None: State name if found, None otherwise
        """
        # List of US states
        states = [
            "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", 
            "Delaware", "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", 
            "Kansas", "Kentucky", "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", 
            "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", 
            "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota", "Ohio", 
            "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota", 
            "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", 
            "Wisconsin", "Wyoming"
        ]
        
        question_lower = question.lower()
        
        # Check for state names in the question
        for state in states:
            if state.lower() in question_lower:
                return state
        
        return None
    
    def _get_general_policy_response(self, question):
        """
        Get a general policy response when no specific state is mentioned
        
        Args:
            question (str): User's policy-related question
        
        Returns:
            str: General policy information
        """
        return """
        Abortion laws and policies vary considerably by state in the United States. Some states have strict restrictions or bans, while others protect access to abortion.

        For the most accurate and up-to-date information about abortion laws in a specific state, please:
        1. Mention the state you're asking about in your question
        2. Visit the Planned Parenthood website (plannedparenthood.org)
        3. Contact a healthcare provider in your area
        4. Call the Planned Parenthood helpline at 1-800-230-PLAN (7526)

        Would you like to know about abortion laws in a specific state?
        """
    
    def _get_state_policy(self, state, question):
        """
        Get state-specific policy information
        
        Args:
            state (str): State name
            question (str): Original question
        
        Returns:
            str: State-specific policy information
        """
        try:
            logger.debug(f"Fetching policy information for state: {state}")
            
            # In a production environment, this would make an actual API call
            # to retrieve the latest policy information from an authoritative source
            
            # Since we're focusing on integration, we'll use a database of policy information
            # This could be expanded with a real API when available
            policy_data = self._get_policy_data_for_state(state)
            
            if not policy_data:
                return f"I don't have specific policy information for {state} at this time. For the most accurate and up-to-date information about abortion laws in {state}, please visit the Planned Parenthood website or contact a healthcare provider in your area."
            
            # Format the response based on policy data
            response = f"""
            Based on the most recent information available for {state}, here is what I can tell you about abortion policies:
            
            {policy_data.get('legal_status', 'Information not available')}
            
            Gestational limits: {policy_data.get('gestational_limits', 'Information not available')}
            
            Restrictions: {policy_data.get('restrictions', 'Information not available')}
            
            Please note that abortion laws can change rapidly, so for the most up-to-date information, I recommend:
            
            1. Contacting Planned Parenthood directly at 1-800-230-PLAN (7526)
            2. Visiting the Planned Parenthood website (plannedparenthood.org)
            3. Consulting with a healthcare provider in {state}
            
            Would you like me to provide information about resources available in {state}?
            """
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting state policy for {state}: {str(e)}", exc_info=True)
            return f"I'm sorry, I'm having trouble retrieving the latest policy information for {state}. For the most accurate information, please consult the Planned Parenthood website or contact a healthcare provider directly."
    
    def _get_policy_data_for_state(self, state):
        """
        Get policy data for a specific state from our database
        
        Args:
            state (str): State name
        
        Returns:
            dict: Policy data for the state
        """
        # This is a simplified database of policy information
        # In a production environment, this would be pulled from an API or database
        policy_database = {
            "California": {
                "legal_status": "Abortion is legal in California. The state has strong protections for abortion access.",
                "gestational_limits": "Abortion is accessible throughout pregnancy when needed to protect the life or health of the pregnant person.",
                "restrictions": "California has expanded access to abortion services and has laws protecting abortion providers and patients."
            },
            "Texas": {
                "legal_status": "Abortion is highly restricted in Texas under Senate Bill 8 (SB 8) and subsequent legislation.",
                "gestational_limits": "Abortion is prohibited after approximately 6 weeks of pregnancy, before many people know they are pregnant.",
                "restrictions": "Texas has numerous restrictions including mandatory waiting periods and parental consent requirements."
            },
            "New York": {
                "legal_status": "Abortion is legal in New York. The state has enacted protections for abortion access.",
                "gestational_limits": "Abortion is accessible up to 24 weeks of pregnancy, and after that if necessary to protect the patient's life or health.",
                "restrictions": "New York has fewer restrictions compared to many states and has expanded access to abortion services."
            },
            "Florida": {
                "legal_status": "Florida has significant restrictions on abortion access.",
                "gestational_limits": "Abortion is prohibited after 15 weeks of pregnancy with limited exceptions.",
                "restrictions": "Florida requires a 24-hour waiting period and parental consent for minors."
            },
            "Illinois": {
                "legal_status": "Abortion is legal in Illinois with statutory protections.",
                "gestational_limits": "Abortion is accessible throughout pregnancy when needed for the health of the pregnant person.",
                "restrictions": "Illinois has removed many restrictions and expanded abortion access."
            },
            "Ohio": {
                "legal_status": "Ohio has significant restrictions on abortion access.",
                "gestational_limits": "Abortion is prohibited after fetal cardiac activity is detected (around 6 weeks) with limited exceptions.",
                "restrictions": "Ohio requires a 24-hour waiting period, parental consent for minors, and has other restrictions."
            },
            "Washington": {
                "legal_status": "Abortion is legal in Washington with statutory protections.",
                "gestational_limits": "Abortion is accessible throughout pregnancy when needed for the health of the pregnant person.",
                "restrictions": "Washington has few restrictions and has enacted laws to protect abortion providers and patients."
            },
            "Massachusetts": {
                "legal_status": "Abortion is legal in Massachusetts with statutory protections.",
                "gestational_limits": "Abortion is accessible up to 24 weeks, and after that if necessary to protect the patient's life or health.",
                "restrictions": "Massachusetts requires parental consent for minors under 16, though there is a judicial bypass option."
            },
            "Michigan": {
                "legal_status": "Abortion is legal in Michigan following a 2022 state constitutional amendment protecting reproductive freedom.",
                "gestational_limits": "Abortion is accessible up to fetal viability, and after that if necessary to protect the patient's life or health.",
                "restrictions": "Michigan has removed many previous restrictions following the constitutional amendment."
            }
        }
        
        return policy_database.get(state, None)
