import json
import time
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ChatbotSimulator:
    """
    Simulate chatbot responses for testing without needing a running server
    """
    def __init__(self):
        self.scores = {}
        self.results = {}
        self.total_score = 0
        self.total_tests = 0
        
        # Define simulated responses for test cases
        self.simulated_responses = {
            "goodbye": {
                "response": "Goodbye! If you have any more questions about reproductive health in the future, don't hesitate to reach out. Take care!",
                "citations": []
            },
            "hello": {
                "response": "Hello! I'm here to help answer your questions about reproductive health. How can I assist you today?",
                "citations": []
            },
            "texas_policy": {
                "response": "In Texas, abortion is prohibited except when necessary to save the pregnant person's life or prevent substantial impairment of major bodily function. There are no exceptions for rape or incest. The ban includes medication abortion, and there are criminal penalties for providers. Texas also has a 'trigger ban' that went into effect after Roe v. Wade was overturned, making abortion a felony. If you need support or more information, I recommend contacting Planned Parenthood or the National Abortion Federation Hotline.",
                "citations": ["Abortion Policy API", "Planned Parenthood"]
            },
            "maine_policy": {
                "response": "In Maine, abortion is legal throughout pregnancy if deemed necessary by a healthcare provider. The state protects abortion rights by statute, ensuring access to care. Abortion is covered by MaineCare (Medicaid) and required to be covered by private insurance. Additionally, Maine allows qualified non-physician healthcare providers to perform abortions. The state has enacted protective shield laws for providers and patients since the Dobbs decision.",
                "citations": ["Abortion Policy API", "Planned Parenthood"]
            },
            "west_virginia_policy": {
                "response": "In West Virginia, abortion is banned with limited exceptions for medical emergencies, rape, and incest (with reporting requirements). The ban includes criminal penalties for providers. Prior abortion restrictions included a 20-week ban, mandatory waiting periods, and parental notification. If you need assistance or additional information, I recommend contacting the National Abortion Federation Hotline for support and resources.",
                "citations": ["Abortion Policy API", "Planned Parenthood"]
            },
            "reproductive_services": {
                "response": "Reproductive healthcare services typically include:\n\n1. Contraception options (birth control pills, IUDs, implants, condoms)\n2. STI/STD testing and treatment\n3. Pregnancy testing and counseling\n4. Prenatal care\n5. Abortion services where legal\n6. Fertility services\n7. Gynecological exams and Pap smears\n8. Breast exams and mammograms\n9. Sexual health education\n10. Menopause management\n\nMost services are available through primary care providers, OB/GYN specialists, Planned Parenthood, community health centers, and specialized reproductive health clinics.",
                "citations": ["Planned Parenthood"]
            },
            "exercise_abortion_pill": {
                "response": "After taking the abortion pill (medical abortion), it's generally recommended to avoid strenuous exercise for 1-2 weeks while your body recovers. Light activity like gentle walking is usually fine after the first few days, but you should listen to your body and rest when needed. Heavy lifting, intense cardio, and vigorous workouts should be avoided until bleeding lessens significantly. Everyone's recovery varies, so it's important to follow your healthcare provider's specific recommendations. If you experience increased bleeding, severe pain, or other concerning symptoms during exercise, stop immediately and contact your healthcare provider.",
                "citations": ["Planned Parenthood"]
            },
            "abortion_pill_effectiveness": {
                "response": "The abortion pill (medical abortion) is highly effective, with a success rate of approximately 94-98% when used before 10 weeks of pregnancy. Effectiveness decreases slightly as pregnancy progresses, which is why it's typically recommended for use within the first 10-11 weeks. In the small percentage of cases where the medication doesn't completely end the pregnancy, a follow-up procedure may be needed. The regimen involves two medications: mifepristone, which blocks progesterone, and misoprostol, which causes uterine contractions. This two-step process is more effective than using misoprostol alone.",
                "citations": ["Planned Parenthood", "World Health Organization"]
            },
            "having_baby": {
                "response": "Congratulations on considering starting a family! Here are some initial steps to prepare for a healthy pregnancy:\n\n1. Start prenatal vitamins with folic acid 2-3 months before trying to conceive\n2. Schedule a preconception check-up with your healthcare provider\n3. Review your medications with your doctor for pregnancy safety\n4. Adopt healthy lifestyle habits (balanced diet, regular exercise, avoid alcohol/smoking)\n5. Track your menstrual cycle to identify fertile windows\n6. Consider genetic counseling if there's family history of genetic conditions\n\nOnce pregnant, early and regular prenatal care is important for monitoring your health and your baby's development. Would you like more specific information about any of these steps?",
                "citations": ["American College of Obstetricians and Gynecologists"]
            },
            "getting_pregnant": {
                "response": "Congratulations on your decision to try to conceive! Here are some steps to help optimize your fertility:\n\n1. Start taking prenatal vitamins with at least 400mcg of folic acid daily (ideally 2-3 months before conception)\n2. Track your menstrual cycle to identify your most fertile days (typically 12-16 days before your next period)\n3. Have regular intercourse every 2-3 days during your fertile window\n4. Maintain a healthy lifestyle - balanced diet, regular exercise, healthy BMI\n5. Avoid smoking, excessive alcohol, and recreational drugs\n6. Schedule a preconception checkup with your healthcare provider\n7. Review any current medications with your doctor\n\nRemember that most healthy couples conceive within one year of trying. If you're under 35 and haven't conceived after a year of trying (or 6 months if you're over 35), consider consulting with a fertility specialist.",
                "citations": ["American College of Obstetricians and Gynecologists", "Centers for Disease Control"]
            },
            "emergency_contraception_period": {
                "response": "Taking emergency contraception (EC) like Plan B can affect when you get your next period. Some common effects include:\n\n- Your period might come earlier or later than expected (typically within a week of the expected date)\n- The flow might be heavier, lighter, or more irregular than usual\n- You might experience spotting before your actual period starts\n\nThese changes are temporary and usually resolve with your next menstrual cycle. If your period is more than a week late after taking emergency contraception, it's advisable to take a pregnancy test. Emergency contraception works primarily by delaying ovulation rather than affecting an existing pregnancy, which is why these menstrual changes occur.",
                "citations": ["Planned Parenthood"]
            }
        }
    
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
            
            # Determine which simulated response to use based on case name and message
            response_key = self._get_response_key(case_name, message)
            
            response_data = self.simulated_responses.get(response_key, {
                "response": "I'm not sure how to respond to that. Could you please provide more information or ask a different question?",
                "citations": []
            })
            
            responses.append({
                "user_message": message,
                "bot_response": response_data["response"],
                "citations": response_data["citations"],
                "status_code": 200
            })
            
            logger.info(f"  Bot response: {response_data['response'][:100]}...")
            
            # Wait between messages for realism
            time.sleep(0.5)
        
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
    
    def _get_response_key(self, case_name, message):
        """
        Map the test case and message to the appropriate simulated response key
        """
        message_lower = message.lower()
        case_name_lower = case_name.lower()
        
        if "goodbye" in message_lower:
            return "goodbye"
        elif "hello" in message_lower or "help" in message_lower:
            return "hello"
        elif "abortion policy" in message_lower:
            if "texas" in message_lower or "texas" in case_name_lower:
                return "texas_policy"
            elif "maine" in message_lower or "maine" in case_name_lower:
                return "maine_policy"
            elif "west virginia" in message_lower or "west virginia" in case_name_lower:
                return "west_virginia_policy"
        elif "services" in message_lower or "services" in case_name_lower:
            return "reproductive_services"
        elif "work out" in message_lower or "exercise" in message_lower:
            return "exercise_abortion_pill"
        elif "effective" in message_lower and "abortion pill" in message_lower:
            return "abortion_pill_effectiveness"
        elif "baby" in message_lower or "baby" in case_name_lower:
            return "having_baby"
        elif "get pregnant" in message_lower or "getting pregnant" in case_name_lower:
            return "getting_pregnant"
        elif "emergency contraception" in message_lower or "period" in message_lower:
            return "emergency_contraception_period"
        
        # Default fallback
        return "hello"
    
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
        
        elif "baby" in case_name.lower() or "getting pregnant" in case_name.lower():
            # Check for supportive response about having a baby or getting pregnant
            combined_response = " ".join([r.get("bot_response", "") for r in responses]).lower()
            pregnancy_keywords = ["prenatal", "pregnancy", "conception", "fertility", "ovulation", "folic acid"]
            mentioned_terms = [k for k in pregnancy_keywords if k in combined_response]
            
            if len(mentioned_terms) >= 3:
                score += 3
                feedback.append(f"Provides detailed pregnancy/fertility information: {', '.join(mentioned_terms[:3])}")
            elif len(mentioned_terms) > 0:
                score += 2
                feedback.append("Provides pregnancy support information")
            else:
                feedback.append("Limited pregnancy/fertility information")
        
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
        
        # Also print a summary to console
        print("\n\nCHATBOT EVALUATION SUMMARY")
        print("=" * 40)
        print(f"Average Score: {results['average_score']:.2f}/10")
        print(f"Total Tests: {results['total_tests']}")
        print("-" * 40)
        print("SCORES BY TEST CASE:")
        for case, score in sorted(results['scores'].items(), key=lambda x: x[1], reverse=True):
            print(f"{case:30s}: {score:.1f}/10")
        print("=" * 40)


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
        "Getting pregnant": ["my boyfriend and i want to get pregnant"],
        "Emergency contraception period": ["Will taking emergency contraception change when I get my period?"]
    }
    
    # Create tester instance
    tester = ChatbotSimulator()
    
    # Run all tests
    results = tester.run_all_tests(test_cases)
    
    # Save report
    tester.save_report(results, "chatbot_simulation_report.json")