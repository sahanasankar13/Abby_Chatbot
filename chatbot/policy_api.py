import os
import logging
import requests
import json
import time
from urllib.parse import quote
from chatbot.gpt_integration import GPTModel

logger = logging.getLogger(__name__)

class PolicyAPI:
    """
    API client for fetching abortion policy information by state using the Abortion Policy API
    """
    def __init__(self):
        """Initialize the Policy API client"""
        logger.info("Initializing Policy API")
        try:
            # Using the provided API key
            self.api_base_url = "https://api.abortionpolicyapi.com/v1"
            self.api_key = os.environ.get("POLICY_API_KEY", "tA3Z3l6l35344")  # Using the provided API key
            self.headers = {"token": self.api_key}
            
            # Initialize GPT model for better response formatting
            self.gpt_model = GPTModel()
            
            # State abbreviation mapping
            self.state_abbreviations = {
                "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
                "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
                "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
                "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
                "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
                "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
                "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
                "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
                "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
                "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
                "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
                "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
                "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC"
            }
            
            # Define available endpoints according to documentation
            self.endpoints = {
                "gestational_limits": f"{self.api_base_url}/gestational_limits",
                "insurance_coverage": f"{self.api_base_url}/insurance_coverage",
                "minors": f"{self.api_base_url}/minors",
                "waiting_periods": f"{self.api_base_url}/waiting_periods"
            }
            
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
        return ("Abortion laws and policies vary considerably by state in the United States. Some states have strict restrictions or bans, "
               "while others protect access to abortion.\n\n"
               "For the most accurate and up-to-date information about abortion laws in a specific state, please:\n"
               "1. Mention the state you're asking about in your question\n"
               "2. Visit the Planned Parenthood website (plannedparenthood.org)\n"
               "3. Contact a healthcare provider in your area\n"
               "4. Call the Planned Parenthood helpline at 1-800-230-PLAN (7526)\n\n"
               "Would you like to know about abortion laws in a specific state?")
    
    def _get_state_policy(self, state, question):
        """
        Get state-specific policy information by calling the Abortion Policy API
        
        Args:
            state (str): State name
            question (str): Original question
        
        Returns:
            str: State-specific policy information
        """
        try:
            logger.debug(f"Fetching policy information for state: {state}")
            
            # Get combined policy data from all endpoints for this state
            policy_data = self._fetch_all_policy_data_for_state(state)
            
            # Log the full policy data to help debug
            logger.debug(f"Combined policy data for {state}: {policy_data}")
            
            if not policy_data:
                return f"I don't have specific policy information for {state} at this time. For the most accurate and up-to-date information about abortion laws in {state}, please visit the Planned Parenthood website or contact a healthcare provider in your area."
            
            # Use GPT to format the response for better readability and to handle multi-queries
            try:
                logger.debug("Using GPT to format policy response")
                gpt_response = self.gpt_model.format_policy_response(question, state, policy_data)
                if gpt_response:
                    logger.debug("Successfully formatted response with GPT")
                    return gpt_response
            except Exception as e:
                logger.error(f"Error formatting with GPT: {str(e)}, falling back to template")
            
            # Fallback to template-based formatting if GPT fails
            logger.debug("Using template-based formatting")
            response = f"Based on the most recent information available for {state}, here is what I can tell you about abortion policies:\n\n"
            
            # Format gestational limits data
            if "gestational_limits" in policy_data:
                gl_data = policy_data["gestational_limits"]
                banned_after_weeks = gl_data.get("banned_after_weeks_since_LMP", "Unknown")
                
                if banned_after_weeks == 99:
                    response += f"Gestational limits: No specific gestational limit identified in {state}.\n\n"
                else:
                    response += f"Gestational limits: Abortion is banned after {banned_after_weeks} weeks since the last menstrual period.\n\n"
                
                # Add exception information
                exceptions = []
                if gl_data.get("exception_life", False):
                    exceptions.append("to save the life of the pregnant person")
                if gl_data.get("exception_health", False):
                    health_type = gl_data.get("exception_health", "")
                    exceptions.append(f"for health reasons ({health_type})")
                if gl_data.get("exception_fetal", False):
                    exceptions.append("for fetal anomalies")
                if gl_data.get("exception_rape_or_incest", False):
                    exceptions.append("in cases of rape or incest")
                
                if exceptions:
                    response += f"Exceptions to these limits include: {', '.join(exceptions)}.\n\n"
            
            # Format waiting period data
            if "waiting_periods" in policy_data:
                wp_data = policy_data["waiting_periods"]
                waiting_hours = wp_data.get("waiting_period_hours", 0)
                counseling_visits = wp_data.get("counseling_visits", 0)
                
                if waiting_hours > 0:
                    response += f"Waiting period: A {waiting_hours}-hour waiting period is required "
                    if counseling_visits > 1:
                        response += f"with {counseling_visits} separate visits needed.\n\n"
                    else:
                        response += "before the procedure.\n\n"
                
                notes = wp_data.get("waiting_period_notes", "")
                if notes:
                    response += f"Note: {notes}\n\n"
            
            # Format minors data
            if "minors" in policy_data:
                minors_data = policy_data["minors"]
                if minors_data.get("parental_consent_required", False) or minors_data.get("parental_notification_required", False):
                    below_age = minors_data.get("below_age", 18)
                    parents_required = minors_data.get("parents_required", 1)
                    
                    response += f"Minors: People below age {below_age} "
                    
                    requirements = []
                    if minors_data.get("parental_consent_required", False):
                        requirements.append(f"consent from {parents_required} parent(s)")
                    if minors_data.get("parental_notification_required", False):
                        requirements.append(f"notification of {parents_required} parent(s)")
                    
                    response += f"require {' and '.join(requirements)}.\n\n"
                    
                    if minors_data.get("judicial_bypass_available", False):
                        response += "A judicial bypass option is available for minors who cannot involve their parents.\n\n"
            
            # Add insurance coverage information
            if "insurance_coverage" in policy_data:
                ins_data = policy_data["insurance_coverage"]
                
                coverage_info = []
                
                # Private insurance
                if ins_data.get("private_coverage_no_restrictions", False):
                    coverage_info.append("Private insurance may cover abortion without restrictions")
                elif "private_exception_life" in ins_data:
                    private_exceptions = []
                    if ins_data.get("private_exception_life", False):
                        private_exceptions.append("life endangerment")
                    if ins_data.get("private_exception_health", False):
                        private_exceptions.append(f"health reasons ({ins_data.get('private_exception_health', '')})")
                    if ins_data.get("private_exception_fetal", False):
                        private_exceptions.append("fetal anomalies")
                    if ins_data.get("private_exception_rape_or_incest", False):
                        private_exceptions.append("rape or incest")
                    
                    if private_exceptions:
                        coverage_info.append(f"Private insurance covers abortion only in cases of: {', '.join(private_exceptions)}")
                
                # Medicaid
                if ins_data.get("medicaid_coverage_provider_patient_decision", False):
                    coverage_info.append("Medicaid may cover abortion based on provider-patient decision")
                elif "medicaid_exception_life" in ins_data:
                    medicaid_exceptions = []
                    if ins_data.get("medicaid_exception_life", False):
                        medicaid_exceptions.append("life endangerment")
                    if ins_data.get("medicaid_exception_health", False):
                        medicaid_exceptions.append(f"health reasons ({ins_data.get('medicaid_exception_health', '')})")
                    if ins_data.get("medicaid_exception_fetal", False):
                        medicaid_exceptions.append("fetal anomalies")
                    if ins_data.get("medicaid_exception_rape_or_incest", False):
                        medicaid_exceptions.append("rape or incest")
                    
                    if medicaid_exceptions:
                        coverage_info.append(f"Medicaid covers abortion only in cases of: {', '.join(medicaid_exceptions)}")
                
                if coverage_info:
                    response += "Insurance coverage:\n"
                    for info in coverage_info:
                        response += f"- {info}\n"
                    response += "\n"
            
            response += "Please note that abortion laws can change rapidly, so for the most up-to-date information, I recommend:\n\n"
            response += "1. Contacting Planned Parenthood directly at 1-800-230-PLAN (7526)\n"
            response += "2. Visiting the Planned Parenthood website (plannedparenthood.org)\n"
            response += "3. Consulting with a healthcare provider in your area\n\n"
            response += "Would you like me to provide more specific information about a particular aspect of abortion access in this state?"
            
            return response
            
        except Exception as e:
            logger.error(f"Error getting state policy for {state}: {str(e)}", exc_info=True)
            return f"I'm sorry, I'm having trouble retrieving the latest policy information for {state}. For the most accurate information, please consult the Planned Parenthood website or contact a healthcare provider directly."
    
    def _fetch_all_policy_data_for_state(self, state):
        """
        Fetch policy data from all API endpoints for a specific state
        
        Args:
            state (str): State name
        
        Returns:
            dict: Combined policy data from all endpoints
        """
        # Get the state abbreviation if it exists in our mapping
        state_abbr = self.state_abbreviations.get(state, state)
        
        try:
            # Make actual API calls to all endpoints
            all_data = {}
            for endpoint_name, endpoint_url in self.endpoints.items():
                try:
                    # Format the URL using state abbreviation as needed by the API
                    state_url = f"{endpoint_url}/states/{state_abbr}"
                    logger.debug(f"Making API request to: {state_url}")
                    
                    # Add a small delay between requests to avoid rate limiting
                    if endpoint_name != list(self.endpoints.keys())[0]:  # Skip delay for first request
                        time.sleep(0.5)
                    
                    # Make the API request
                    response = requests.get(state_url, headers=self.headers)
                    
                    if response.status_code == 200:
                        data = response.json()
                        # Log the actual API response
                        logger.debug(f"API response for {endpoint_name}: {data}")
                        
                        # API returns data keyed by state, so we extract just that state's data
                        if state_abbr in data:
                            all_data[endpoint_name] = data[state_abbr]
                            logger.debug(f"Found data for state abbr {state_abbr}: {data[state_abbr]}")
                        elif state in data:
                            all_data[endpoint_name] = data[state]
                            logger.debug(f"Found data for state name {state}: {data[state]}")
                        else:
                            logger.warning(f"State {state} not found in response data: {data.keys()}")
                            all_data[endpoint_name] = {}
                    else:
                        logger.warning(f"API returned status code {response.status_code} for {endpoint_name}")
                        all_data[endpoint_name] = {}
                except Exception as e:
                    logger.error(f"Error fetching {endpoint_name} data for {state}: {str(e)}")
                    all_data[endpoint_name] = {}
            
            # If we have some data, return it
            if any(all_data.values()):
                return all_data
            
            # Fall back to static database if API calls failed or returned no data
            logger.info(f"No data from API for {state}, falling back to static database")
            return self._get_policy_data_for_state(state)
        
        except Exception as e:
            logger.error(f"Error in API calls for {state}: {str(e)}", exc_info=True)
            # Fall back to static database
            logger.info(f"Error in API calls for {state}, falling back to static database")
            return self._get_policy_data_for_state(state)
    
    def _get_policy_data_for_state(self, state):
        """
        Get policy data for a specific state from our database
        
        Args:
            state (str): State name
        
        Returns:
            dict: Policy data for the state formatted like the API response
        """
        # This is a simulated database that models the real API responses
        # In production, this would be replaced with actual API calls
        static_database = {
            "California": {
                "gestational_limits": {
                    "banned_after_weeks_since_LMP": 99,
                    "exception_life": True,
                    "exception_health": "Any",
                    "exception_fetal": True,
                    "exception_rape_or_incest": True,
                    "no_restrictions": True
                },
                "waiting_periods": {},
                "minors": {
                    "below_age": 18,
                    "parental_consent_required": False,
                    "parents_required": 0,
                    "judicial_bypass_available": True,
                    "allows_minor_to_consent": True
                },
                "insurance_coverage": {
                    "requires_coverage": True,
                    "private_coverage_no_restrictions": True,
                    "exchange_coverage_no_restrictions": True,
                    "medicaid_coverage_provider_patient_decision": True
                }
            },
            "Texas": {
                "gestational_limits": {
                    "banned_after_weeks_since_LMP": 6,
                    "exception_life": True,
                    "exception_health": "Major Bodily Function",
                    "exception_fetal": False,
                    "exception_rape_or_incest": False
                },
                "waiting_periods": {
                    "waiting_period_hours": 24,
                    "counseling_visits": 2,
                    "waiting_period_notes": "An ultrasound must be performed at least 24 hours before the abortion."
                },
                "minors": {
                    "below_age": 18,
                    "parental_consent_required": True,
                    "parental_notification_required": True,
                    "parents_required": 1,
                    "judicial_bypass_available": True
                },
                "insurance_coverage": {
                    "private_exception_life": True,
                    "private_exception_health": "Major Bodily Function",
                    "exchange_exception_life": True,
                    "exchange_exception_health": "Major Bodily Function",
                    "medicaid_exception_life": True,
                    "medicaid_exception_rape_or_incest": True
                }
            },
            "New York": {
                "gestational_limits": {
                    "banned_after_weeks_since_LMP": 24,
                    "exception_life": True,
                    "exception_health": "Any",
                    "exception_fetal": True
                },
                "waiting_periods": {},
                "minors": {
                    "allows_minor_to_consent": True
                },
                "insurance_coverage": {
                    "private_coverage_no_restrictions": True,
                    "exchange_coverage_no_restrictions": True,
                    "medicaid_coverage_provider_patient_decision": True
                }
            },
            "Florida": {
                "gestational_limits": {
                    "banned_after_weeks_since_LMP": 15,
                    "exception_life": True,
                    "exception_health": "Major Bodily Function",
                    "exception_fetal": True,
                    "exception_rape_or_incest": True
                },
                "waiting_periods": {
                    "waiting_period_hours": 24,
                    "counseling_visits": 2
                },
                "minors": {
                    "below_age": 18,
                    "parental_consent_required": True,
                    "parents_required": 1,
                    "judicial_bypass_available": True
                },
                "insurance_coverage": {
                    "exchange_exception_life": True,
                    "exchange_exception_rape_or_incest": True,
                    "medicaid_exception_life": True,
                    "medicaid_exception_rape_or_incest": True,
                    "private_coverage_no_restrictions": True
                }
            }
        }
        
        return static_database.get(state, None)
