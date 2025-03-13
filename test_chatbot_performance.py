import requests
import json
import time
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotTester:
    """
    Test the chatbot with different use cases and provide scoring
    """
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.chat_endpoint = f"{base_url}/api/chat"
        self.scores = {}
        self.results = {}
        self.total_score = 0
        self.total_tests = 0
    
    def test_case(self, case_name, messages):
        """
        Test a specific use case consisting of one or more messages
        
        Args:
            case_name (str): Name of the test case
            messages (list): List of messages to send in sequence
        
        Returns:
            dict: Test results
        """
        logger.info(f"Testing case: {case_name}")
        
        responses = []
        session_start = time.time()
        
        for i, message in enumerate(messages):
            logger.info(f"  Message {i+1}: {message}")
            
            try:
                # Send message to chatbot
                response = requests.post(
                    self.chat_endpoint,
                    json={"message": message},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    bot_response = data.get('response', '')
                    citations = data.get('citations', [])
                    
                    responses.append({
                        "user_message": message,
                        "bot_response": bot_response,
                        "citations": citations,
                        "status_code": response.status_code
                    })
                    
                    logger.info(f"  Bot response: {bot_response[:100]}...")
                else:
                    error_text = f"Error: Status code {response.status_code}"
                    try:
                        error_text += f", {response.json().get('error', '')}"
                    except:
                        error_text += f", {response.text[:100]}"
                    
                    responses.append({
                        "user_message": message,
                        "bot_response": error_text,
                        "status_code": response.status_code
                    })
                    logger.error(f"  {error_text}")
            
            except Exception as e:
                error_text = f"Exception: {str(e)}"
                responses.append({
                    "user_message": message,
                    "bot_response": error_text,
                    "status_code": 500
                })
                logger.error(f"  {error_text}")
            
            # Wait between messages
            time.sleep(1)
        
        session_duration = time.time() - session_start
        
        # Evaluate the responses
        score, feedback = self.evaluate_responses(case_name, responses)
        
        self.total_score += score
        self.total_tests += 1
        
        result = {
            "case_name": case_name,
            "responses": responses,
            "score": score,
            "feedback": feedback,
            "duration": session_duration
        }
        
        self.results[case_name] = result
        self.scores[case_name] = score
        
        logger.info(f"  Score: {score}/10 - {feedback}")
        logger.info("-" * 50)
        
        return result
    
    def evaluate_responses(self, case_name, responses):
        """
        Evaluate the quality of responses for a test case
        
        Args:
            case_name (str): Name of the test case
            responses (list): List of response data
        
        Returns:
            tuple: (score, feedback)
        """
        # Default score is 5/10
        score = 5
        feedback = []
        
        # Check if all requests were successful
        all_successful = all(r.get("status_code") == 200 for r in responses)
        
        if not all_successful:
            score -= 3
            feedback.append("Some requests failed")
        
        # Look for empty or very short responses
        has_empty = any(not r.get("bot_response", "") for r in responses)
        has_short = any(len(r.get("bot_response", "")) < 20 for r in responses)
        
        if has_empty:
            score -= 2
            feedback.append("Empty responses detected")
        elif has_short:
            score -= 1
            feedback.append("Very short responses detected")
        
        # Look for error messages in the responses
        has_errors = any("error" in r.get("bot_response", "").lower() for r in responses)
        
        if has_errors:
            score -= 1
            feedback.append("Error messages in responses")
        
        # Case-specific scoring
        if "goodbye" in case_name.lower():
            # Check for friendly goodbye response
            last_response = responses[-1]["bot_response"].lower()
            if "bye" in last_response or "goodbye" in last_response or "take care" in last_response:
                score += 2
                feedback.append("Appropriate goodbye response")
            else:
                feedback.append("Missing goodbye acknowledgment")
        
        elif "hello" in case_name.lower():
            # Check for greeting
            first_response = responses[-1]["bot_response"].lower()
            if "hello" in first_response or "hi" in first_response or "welcome" in first_response:
                score += 2
                feedback.append("Appropriate greeting")
            else:
                feedback.append("Missing greeting")
        
        elif "policy" in case_name.lower():
            last_response = responses[-1]["bot_response"].lower()
            if "policy" in last_response or "law" in last_response or "legal" in last_response:
                score += 2
                feedback.append("Contains policy information")
            else:
                feedback.append("Missing policy information")
            
            # Check for state name in the response
            state_name = case_name.split("in ")[-1].lower() if "in " in case_name else ""
            if state_name and state_name in last_response:
                score += 1
                feedback.append(f"Response mentions {state_name}")
            elif state_name:
                feedback.append(f"Response does not mention {state_name}")
            
            # Check for citations when discussing policy
            has_citations = any(len(r.get("citations", [])) > 0 for r in responses)
            if has_citations:
                score += 1
                feedback.append("Includes citations")
            else:
                feedback.append("Missing citations")
        
        elif "services" in case_name.lower():
            # Check for comprehensive information
            combined_response = " ".join([r.get("bot_response", "") for r in responses]).lower()
            service_keywords = ["birth control", "contraception", "testing", "abortion", "pregnancy"]
            mentioned_services = [k for k in service_keywords if k in combined_response]
            
            if len(mentioned_services) >= 3:
                score += 3
                feedback.append(f"Mentions multiple services: {', '.join(mentioned_services)}")
            elif len(mentioned_services) >= 1:
                score += 1
                feedback.append(f"Mentions some services: {', '.join(mentioned_services)}")
            else:
                feedback.append("Few specific services mentioned")
        
        elif "abortion pill" in case_name.lower():
            # Check for specific information about the abortion pill
            combined_response = " ".join([r.get("bot_response", "") for r in responses]).lower()
            pill_keywords = ["mifepristone", "misoprostol", "medical abortion"]
            mentioned_terms = [k for k in pill_keywords if k in combined_response]
            
            if "work out" in case_name.lower() and "exercise" in combined_response:
                score += 3
                feedback.append("Addresses exercise after abortion pill")
            elif "effective" in case_name.lower() and any(term in combined_response for term in ["percent", "success rate", "effective"]):
                score += 3
                feedback.append("Addresses effectiveness of abortion pill")
            elif len(mentioned_terms) > 0:
                score += 2
                feedback.append(f"Mentions specific terms: {', '.join(mentioned_terms)}")
            
            # Check for citations
            has_citations = any(len(r.get("citations", [])) > 0 for r in responses)
            if has_citations:
                score += 1
                feedback.append("Includes citations")
        
        elif "baby" in case_name.lower():
            # Check for supportive response about having a baby
            combined_response = " ".join([r.get("bot_response", "") for r in responses]).lower()
            if "prenatal" in combined_response or "pregnancy" in combined_response:
                score += 3
                feedback.append("Provides pregnancy support information")
            else:
                feedback.append("Limited pregnancy support information")
        
        elif "emergency contraception" in case_name.lower() or "period" in case_name.lower():
            # Check for accurate information about emergency contraception
            combined_response = " ".join([r.get("bot_response", "") for r in responses]).lower()
            ec_keywords = ["plan b", "morning after", "levonorgestrel", "menstrual cycle", "period"]
            mentioned_terms = [k for k in ec_keywords if k in combined_response]
            
            if len(mentioned_terms) >= 2:
                score += 3
                feedback.append(f"Provides detailed EC information: {', '.join(mentioned_terms)}")
            elif len(mentioned_terms) == 1:
                score += 1
                feedback.append(f"Mentions {mentioned_terms[0]}")
            else:
                feedback.append("Limited EC information")
        
        # Cap score between 0 and 10
        score = max(0, min(10, score))
        
        # Generate overall feedback
        if score >= 9:
            overall = "Excellent response"
        elif score >= 7:
            overall = "Good response"
        elif score >= 5:
            overall = "Adequate response"
        elif score >= 3:
            overall = "Poor response"
        else:
            overall = "Inadequate response"
        
        feedback.insert(0, overall)
        
        return score, "; ".join(feedback)
    
    def run_all_tests(self, test_cases):
        """
        Run all test cases and generate a report
        
        Args:
            test_cases (dict): Dictionary of test cases
        
        Returns:
            dict: Test results
        """
        start_time = time.time()
        
        for case_name, messages in test_cases.items():
            self.test_case(case_name, messages)
        
        total_time = time.time() - start_time
        avg_score = self.total_score / self.total_tests if self.total_tests > 0 else 0
        
        logger.info(f"All tests completed. Average score: {avg_score:.2f}/10")
        logger.info(f"Total time: {total_time:.2f} seconds")
        
        return {
            "average_score": avg_score,
            "total_tests": self.total_tests,
            "total_time": total_time,
            "scores": self.scores,
            "detailed_results": self.results
        }
    
    def save_report(self, results, filename=None):
        """
        Save test results to a JSON file
        
        Args:
            results (dict): Test results
            filename (str, optional): Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"chatbot_test_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Test report saved to {filename}")


if __name__ == "__main__":
    # Define test cases from the user's requirements
    test_cases = {
        "Goodbye": ["Goodbye"],
        "Hello with typo": ["Helllo, can yu help me?"],
        "Personal info and Texas policy": [
            "My name is Chloe Nicole and my email is chloe@sahana.com. What is the abortion policy in Texas?"
        ],
        "Maine policy": ["What is the abortion policy in Maine?"],
        "Texas policy": ["What is the abortion policy in Texas?"],
        "West Virginia policy": ["What is the abortion policy in West Virginia?"],
        "Reproductive healthcare services": ["What reproductive healthcare services are available?"],
        "Exercise after abortion pill": ["Can I work out after I take the abortion pill?"],
        "Abortion pill effectiveness": ["How effective is the abortion pill?"],
        "Having a baby": ["My boyfriend and I want to have a baby"],
        "Emergency contraception period": ["Will taking emergency contraception change when I get my period?"]
    }
    
    # Create tester instance
    tester = ChatbotTester()
    
    # Run all tests
    results = tester.run_all_tests(test_cases)
    
    # Save report
    tester.save_report(results)