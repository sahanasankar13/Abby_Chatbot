import os
import logging
import requests
import json
import re
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)


class PolicyAPI:
    """
    Integration with the abortion policy API to provide up-to-date policy information
    """

    STATE_NAMES = {
        "AL": "Alabama",
        "AK": "Alaska",
        "AZ": "Arizona",
        "AR": "Arkansas",
        "CA": "California",
        "CO": "Colorado",
        "CT": "Connecticut",
        "DE": "Delaware",
        "FL": "Florida",
        "GA": "Georgia",
        "HI": "Hawaii",
        "ID": "Idaho",
        "IL": "Illinois",
        "IN": "Indiana",
        "IA": "Iowa",
        "KS": "Kansas",
        "KY": "Kentucky",
        "LA": "Louisiana",
        "ME": "Maine",
        "MD": "Maryland",
        "MA": "Massachusetts",
        "MI": "Michigan",
        "MN": "Minnesota",
        "MS": "Mississippi",
        "MO": "Missouri",
        "MT": "Montana",
        "NE": "Nebraska",
        "NV": "Nevada",
        "NH": "New Hampshire",
        "NJ": "New Jersey",
        "NM": "New Mexico",
        "NY": "New York",
        "NC": "North Carolina",
        "ND": "North Dakota",
        "OH": "Ohio",
        "OK": "Oklahoma",
        "OR": "Oregon",
        "PA": "Pennsylvania",
        "RI": "Rhode Island",
        "SC": "South Carolina",
        "SD": "South Dakota",
        "TN": "Tennessee",
        "TX": "Texas",
        "UT": "Utah",
        "VT": "Vermont",
        "VA": "Virginia",
        "WA": "Washington",
        "WV": "West Virginia",
        "WI": "Wisconsin",
        "WY": "Wyoming",
        "DC": "District of Columbia"
    }

    # Case-insensitive lookup: "california" -> "CA"
    STATE_NAMES_LOWER = {
        name.lower(): code
        for code, name in STATE_NAMES.items()
    }

    def __init__(self):
        """Initialize the Policy API client"""
        logger.info("Initializing Policy API")
        # Check env vars for API key
        self.api_key = os.environ.get(
            "ABORTION_POLICY_API_KEY") or os.environ.get("POLICY_API_KEY")
        self.base_url = "https://api.abortionpolicyapi.com/v1"

        if not self.api_key:
            logger.warning(
                "Abortion Policy API key not found in environment variables")
        else:
            logger.info("Abortion Policy API key found")

        # We'll lazy-load GPT if needed
        self.gpt_model = None
        logger.info("Policy API initialized successfully")

    def _extract_state_from_question(self, question: str) -> Optional[str]:
        """
        Extract state information from a user's question. 
        Return the 2-letter state code if found, else None.
        """
        question_lower = question.lower().strip()

        # Quick pass: check exact name matches (e.g. "california")
        for state_name_lower, code in self.STATE_NAMES_LOWER.items():
            if state_name_lower in question_lower:
                logger.debug(
                    f"Found state name '{state_name_lower}' in question -> code={code}"
                )
                return code

        # Check for known 2-letter abbreviations in the text
        # (Only valid if they match an actual state code)
        pattern = r'\b([A-Za-z]{2})\b'
        matches = re.findall(pattern, question)
        for match in matches:
            abbr = match.upper()
            if abbr in self.STATE_NAMES:
                logger.debug(f"Found state abbreviation '{abbr}' in question")
                return abbr

        # If nothing found, return None
        logger.debug("No valid US state recognized in the question")
        return None

    def get_policy_data(self, state_code: str) -> Dict[str, Any]:
        """
        Get abortion policy data for a specific state from the Abortion Policy API
        """
        if not self.api_key:
            logger.error("API key not found when trying to fetch policy data")
            return {"error": "API key not configured"}

        # Verify that state_code is indeed in our known states
        if state_code.upper() not in self.STATE_NAMES:
            logger.warning(
                f"Attempted to fetch data for invalid state code '{state_code}'"
            )
            return {"error": f"Not a recognized US state code: {state_code}"}

        try:
            # Define endpoints
            endpoints = {
                "waiting_periods": "waiting_periods",
                "insurance_coverage": "insurance_coverage",
                "gestational_limits": "gestational_limits",
                "minors": "minors"
            }

            policy_data = {
                "state_code": state_code.upper(),
                "state_name": self.STATE_NAMES[state_code.upper()],
                "endpoints": {}
            }

            headers = {"token": self.api_key}

            import time
            for key, endpoint in endpoints.items():
                url = f"{self.base_url}/{endpoint}/states/{state_code.upper()}"
                logger.debug(f"Making request to {url}")
                time.sleep(0.4)  # slight delay to avoid rate-limits

                response = requests.get(url, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        policy_data["endpoints"][key] = data
                        policy_data["success"] = True
                else:
                    logger.warning(
                        f"Failed endpoint {key} with status {response.status_code}"
                    )
                    policy_data["endpoints"][key] = {
                        "error": f"Status code {response.status_code}"
                    }

            # If no successful data at all
            if not policy_data.get("success"):
                policy_data[
                    "error"] = "No policy data retrieved from any endpoint"
            return policy_data

        except Exception as e:
            logger.error(f"Error fetching policy data: {str(e)}",
                         exc_info=True)
            return {"error": str(e)}

    def get_policy_response(self,
                            question: str,
                            conversation_history: Optional[List[Dict[
                                str, Any]]] = None,
                            location_context: Optional[str] = None) -> str:
        """
        Process a policy-related question and return a response:
         - If no valid US state is found, we disclaim that we only have US data.
         - Otherwise, fetch data from the API and format the response.
        """
        # 1) Check location_context first
        state_code = None
        if location_context:
            # If location_context is recognized in our state map
            if location_context.lower() in self.STATE_NAMES_LOWER:
                state_code = self.STATE_NAMES_LOWER[location_context.lower()]
                logger.info(
                    f"Converted location context '{location_context}' -> state code '{state_code}'"
                )

        # 2) If still None, try extracting from question
        if not state_code:
            extracted_code = self._extract_state_from_question(question)
            if extracted_code:
                state_code = extracted_code
                logger.info(
                    f"Extracted state code from question: {state_code}")

        # 3) If still None, we disclaim we only have US data, but provide more helpful response
        if not state_code:
            logger.debug(
                "No valid US state recognized -> returning enhanced international response.")
            
            # Check if the question mentions a specific non-US country
            non_us_countries = ['india', 'canada', 'uk', 'australia', 'mexico', 'france', 'germany', 
                               'china', 'japan', 'brazil', 'spain', 'italy', 'russia', 'north korea']
            
            mentioned_country = None
            for country in non_us_countries:
                if country in question.lower():
                    mentioned_country = country
                    break
            
            if mentioned_country:
                return (
                    f"I understand you're asking about abortion access in {mentioned_country.title()}. "
                    f"While I have detailed policy information only for U.S. states, I recommend consulting "
                    f"local healthcare providers or reproductive health organizations in {mentioned_country.title()} "
                    f"for accurate, up-to-date information. International organizations like the World Health "
                    f"Organization (WHO) may also provide relevant resources."
                )
            else:
                return (
                    "I'm sorry, but I only have detailed policy information for U.S. states right now. "
                    "For information about abortion policies in other countries, I recommend consulting "
                    "local healthcare providers or international organizations like the World Health Organization (WHO). "
                    "Would you like me to provide information about a specific U.S. state instead?")

        # 4) Fetch policy data
        policy_data = self.get_policy_data(state_code)
        if "error" in policy_data:
            logger.warning(
                f"Error in policy data for state '{state_code}': {policy_data['error']}"
            )
            return (
                f"I'm sorry, I'm having trouble accessing policy information for {state_code}. "
                "Abortion policies vary by state and may change. "
                "You might consider contacting Planned Parenthood or a local provider for the most current details."
            )

        # 5) Format policy data into a user-friendly response
        return self._format_policy_response(question, state_code, policy_data)

    def _format_policy_response(self, question: str, state_code: str,
                                policy_data: Dict[str, Any]) -> str:
        """
        Format policy data into a user-friendly, conversational response.
        This method sanitizes incorrect data and formats the response in a clear, accurate way.
        """
        # import GPTModel dynamically to avoid circular imports
        from chatbot.gpt_integration import GPTModel
        from chatbot.citation_manager import CitationManager

        if not self.gpt_model:
            self.gpt_model = GPTModel()

        state_name = policy_data["state_name"]

        # Sanitize policy data to fix known issues
        if "endpoints" in policy_data and "gestational_limits" in policy_data[
                "endpoints"]:
            gestational_data = policy_data["endpoints"]["gestational_limits"]

            # Fix the 99 weeks error
            if isinstance(
                    gestational_data, dict
            ) and "banned_after_weeks_since_LMP" in gestational_data:
                if gestational_data["banned_after_weeks_since_LMP"] == 99:
                    gestational_data[
                        "banned_after_weeks_since_LMP"] = "no specific limit"
                    gestational_data[
                        "limit_type"] = "No specific gestational limit"

        # Create sanitized policy JSON
        policy_json = json.dumps(policy_data, indent=2)

        # Better prompt for GPT to create a more accurate, concise response
        policy_prompt = f"""
        The user asked: "{question}"

        We have abortion policy data for {state_name} from the Abortion Policy API:
        {policy_json}

        Provide a brief, accurate response about abortion access in {state_name}:

        1. Start with a direct answer: "Yes, abortion is available in {state_name}." or "No, abortion is not available in {state_name}."
        2. IMPORTANT: If the data shows a "banned_after_weeks_since_LMP" value of 99 or "no specific limit", say there is "no specific gestational             limit" rather than using the number.
        3. Include key policy information in 1-2 sentences. Do NOT include fictional details if data is missing.
        4. Keep your response under 5 sentences total, focusing on the most important information.
        5. Use simple, clear language without jargon.
        6. End with "(Source: Abortion Policy API)" only if you used real data from the API.

        Do NOT mention 99 weeks or any impossible medical values. A normal pregnancy is only about 40 weeks.
        """

        try:
            response_text = self.gpt_model.get_response(policy_prompt)

            # Additional safety check: Replace any mention of "99 weeks" if it somehow slipped through
            response_text = response_text.replace(
                "99 weeks", "no specific gestational limit")

            # Verify the response doesn't contain medically impossible values
            if "week" in response_text:
                # Check for any number before "week" that's greater than 40
                week_numbers = re.findall(r'(\d+)\s*weeks?', response_text)
                for num in week_numbers:
                    if int(num) > 40:
                        # Replace with a reasonable value
                        response_text = response_text.replace(
                            f"{num} weeks", "no specific gestational limit")

            return response_text

        except Exception as e:
            logger.error(f"Error formatting policy response: {str(e)}",
                         exc_info=True)
            # Fallback: Basic text without GPT
            fallback = (
                f"Abortion is available in {state_name}, but I'm having trouble providing specific details right now. "
                f"Please check with a healthcare provider for current policy information as regulations can change. "
                "(Source: Abortion Policy API)")

            return fallback

    def get_state_code(self, location_context: str) -> Optional[str]:
        """Converts a location string (state name or abbreviation) into a 2-letter state code."""
        location_context = location_context.lower()
        if location_context in self.STATE_NAMES_LOWER:
            return self.STATE_NAMES_LOWER[location_context]
        
        # Check for known 2-letter abbreviations in the text
        # (Only valid if they match an actual state code)
        pattern = r'\b([A-Za-z]{2})\b'
        matches = re.findall(pattern, location_context)
        for match in matches:
            abbr = match.upper()
            if abbr in self.STATE_NAMES:
                return abbr
        return None

    def get_abortion_policy(self, location_context: str) -> str:
        """
        Get abortion policy information for a given location

        Args:
            location_context (str): The location to get policy for (state name or abbreviation)

        Returns:
            str: Policy information for the location
        """
        try:
            # Check if this is a non-US country
            if location_context.lower() in ['india', 'canada', 'uk', 'australia', 'mexico', 'france', 'germany', 
                                         'china', 'japan', 'brazil', 'spain', 'italy', 'russia', 'north korea']:
                logger.debug(f"Non-US country detected: {location_context}")
                # Return response WITHOUT Abortion Policy API citation marker
                # This ensures the citation manager won't add the wrong source
                return (
                    f"I'm sorry, but I don't have specific information about abortion access in {location_context.title()}. "
                    f"It's important to note that different countries have different regulations regarding reproductive healthcare. "
                    f"For accurate, up-to-date information, I recommend consulting local healthcare providers or international "
                    f"organizations like the World Health Organization (WHO) or the United Nations Population Fund that specialize in "
                    f"reproductive rights. Remember, prioritizing your safety and well-being is crucial.")

            # Convert to state code for API call
            state_code = self.get_state_code(location_context)

            # If no valid US state is found, we disclaim we only have US data
            if not state_code:
                logger.debug("No valid US state recognized -> returning fallback message.")
                return (
                    "I'm sorry, but I only have policy information for U.S. states right now. "
                    "I don't have data about abortion policies in that location.")

            # Fetch policy data
            policy_data = self.get_policy_data(state_code)
            if "error" in policy_data:
                logger.warning(
                    f"Error in policy data for state '{state_code}': {policy_data['error']}"
                )
                return (
                    f"I'm sorry, I'm having trouble accessing policy information for {state_code}. "
                    "Abortion policies vary by state and may change. "
                    "You might consider contacting Planned Parenthood or a local provider for the most current details."
                )

            # Format policy data into a user-friendly response
            return self._format_policy_response(location_context, state_code, policy_data)


        except Exception as e:
            logger.error(f"Error getting abortion policy: {e}", exc_info=True)
            return "I'm sorry, something went wrong. Please try again later."