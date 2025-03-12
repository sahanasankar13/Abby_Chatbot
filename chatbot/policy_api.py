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
        import re

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

        # 3) If still None, we disclaim we only have US data
        if not state_code:
            logger.debug(
                "No valid US state recognized -> returning fallback message.")
            return (
                "I'm sorry, but I only have policy information for U.S. states right now. "
                "I don't have data about abortion policies in that location.")

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
        This is where you instruct GPT to create the final text. 
        """
        # import GPTModel dynamically to avoid circular imports
        from chatbot.gpt_integration import GPTModel
        from chatbot.citation_manager import CitationManager

        if not self.gpt_model:
            self.gpt_model = GPTModel()

        state_name = policy_data["state_name"]
        policy_json = json.dumps(policy_data, indent=2)

        # Prompt: keep it conversational, no headings, no bullet points
        policy_prompt = f"""
        The user asked: "{question}"

        We have abortion policy data for {state_name} from the Abortion Policy API:
        {policy_json}

        Please provide a short, conversational overview. 
        - Do NOT use headings or bullet points.
        - Speak naturally in short paragraphs.
        - End with "(Source: Abortion Policy API)" only if you use the data. 
        - If you see no relevant data, do NOT mention the source. 
        - Emphasize that laws can change.
        """

        try:
            response_text = self.gpt_model.get_response(policy_prompt)

            # Optionally add citation marker if data was used
            # But if your GPT prompt already adds it, you may omit this step.
            citation_mgr = CitationManager()
            # Add the actual marker to track for your UI, if desired
            final_text = citation_mgr.add_citation_to_text(
                response_text, "abortion_policy_api")
            return final_text

        except Exception as e:
            logger.error(f"Error formatting policy response: {str(e)}",
                         exc_info=True)
            # Fallback: Basic text without GPT
            fallback = (
                f"Here's what I know about abortion policy in {state_name}, but I'm having trouble "
                "formatting the data right now. These policies can change, so please check for updates. "
                "(Source: Abortion Policy API)")
            citation_mgr = CitationManager()
            return citation_mgr.add_citation_to_text(fallback,
                                                     "abortion_policy_api")
