import os
import logging
import requests
import time
import json
from functools import lru_cache
from chatbot.gpt_integration import GPTModel

logger = logging.getLogger(__name__)

class PolicyAPI:
    """
    Interface to abortion policy API for state-specific legal information
    with caching for better performance
    """
    def __init__(self):
        """Initialize the policy API interface"""
        logger.info("Initializing Policy API")
        try:
            self.base_url = "https://api.abortionpolicyapi.com/v1"
            self.api_key = os.environ.get('ABORTION_POLICY_API_KEY')

            if not self.api_key:
                logger.warning("Abortion Policy API key not found in environment variables")

            # Initialize GPT to help format responses
            self.gpt_formatter = GPTModel()

            # Initialize cache
            self.cache = {}
            self.cache_expiry = 24 * 60 * 60  # 24 hours in seconds

            logger.info("Policy API initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Policy API: {str(e)}", exc_info=True)
            raise

    def get_policy_response(self, question):
        """
        Get a response for policy-related questions with caching

        Args:
            question (str): User's question

        Returns:
            str: Response with policy information
        """
        try:
            # Extract state from question
            state = self.extract_state(question)

            if not state:
                return self.gpt_formatter.get_response(f"""
                The user asked a policy-related question, but I couldn't determine which state they're asking about.
                Question: {question}

                Please ask them to specify which state they're interested in information about.
                """)

            # Check cache first
            cached_data = self.get_from_cache(f"state_policy_{state}")

            if cached_data:
                logger.debug(f"Using cached policy data for {state}")
                policy_data = cached_data
            else:
                # Get policy data for the state
                policy_data = self.get_state_policy(state)

                # Save to cache
                if policy_data:
                    self.save_to_cache(f"state_policy_{state}", policy_data)

            if not policy_data:
                return self.gpt_formatter.get_response(f"""
                The user asked about abortion policy in {state}, but I couldn't retrieve policy data.
                Question: {question}

                Please apologize and explain that we currently don't have data available for {state}.
                """)

            # Format the response using GPT
            return self.gpt_formatter.format_policy_response(question, state, policy_data)

        except Exception as e:
            logger.error(f"Error getting policy response: {str(e)}", exc_info=True)
            return "I'm sorry, I encountered an error retrieving policy information. Please try asking in a different way or about a different state."

    def get_from_cache(self, key):
        """
        Get data from cache if not expired

        Args:
            key (str): Cache key

        Returns:
            dict or None: Cached data or None if not found or expired
        """
        try:
            if key in self.cache:
                cache_entry = self.cache[key]
                # Check if cache entry is still valid
                if time.time() - cache_entry['timestamp'] < self.cache_expiry:
                    logger.debug(f"Cache hit for {key}")
                    return cache_entry['data']
                else:
                    logger.debug(f"Cache expired for {key}")
                    del self.cache[key]
            return None
        except Exception as e:
            logger.error(f"Error retrieving from cache: {str(e)}")
            return None

    def save_to_cache(self, key, data):
        """
        Save data to cache

        Args:
            key (str): Cache key
            data (dict): Data to cache
        """
        try:
            self.cache[key] = {
                'data': data,
                'timestamp': time.time()
            }
            logger.debug(f"Saved to cache: {key}")
        except Exception as e:
            logger.error(f"Error saving to cache: {str(e)}")


    def extract_state(self, question):
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
            "Wisconsin", "Wyoming", "District of Columbia"
        ]

        question_lower = question.lower()

        # Check for state names in the question
        for state in states:
            if state.lower() in question_lower:
                return state

        return None

    def get_state_policy(self, state):
        """
        Get state-specific policy information by calling the Abortion Policy API

        Args:
            state (str): State name

        Returns:
            dict: State-specific policy information or None if not found
        """
        try:
            logger.debug(f"Fetching policy information for state: {state}")
            # Get combined policy data from all endpoints for this state
            policy_data = self._fetch_all_policy_data_for_state(state)

            # Log the full policy data to help debug
            logger.debug(f"Combined policy data for {state}: {policy_data}")

            return policy_data

        except Exception as e:
            logger.error(f"Error getting state policy for {state}: {str(e)}", exc_info=True)
            return None


    def _fetch_all_policy_data_for_state(self, state):
        """
        Fetch policy data from all API endpoints for a specific state

        Args:
            state (str): State name

        Returns:
            dict: Combined policy data from all endpoints
        """
        # Get the state abbreviation if it exists in our mapping
        state_abbreviations = {
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
        state_abbr = state_abbreviations.get(state, state)

        try:
            # Make actual API calls to all endpoints
            all_data = {}
            endpoints = {
                "gestational_limits": f"{self.base_url}/gestational_limits",
                "insurance_coverage": f"{self.base_url}/insurance_coverage",
                "minors": f"{self.base_url}/minors",
                "waiting_periods": f"{self.base_url}/waiting_periods"
            }
            for endpoint_name, endpoint_url in endpoints.items():
                try:
                    # Format the URL using state abbreviation as needed by the API
                    state_url = f"{endpoint_url}/states/{state_abbr}"
                    logger.debug(f"Making API request to: {state_url}")

                    # Add a small delay between requests to avoid rate limiting
                    if endpoint_name != list(endpoints.keys())[0]:  # Skip delay for first request
                        time.sleep(0.5)

                    # Make the API request
                    response = requests.get(state_url, headers={"token": self.api_key})

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