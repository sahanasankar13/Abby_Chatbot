import os
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from openai import OpenAI
from utils.metrics import increment_counter, record_time, record_api_call

logger = logging.getLogger(__name__)

# Set up file logging
file_handler = logging.FileHandler('evaluation_logs.json', mode='a')
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

class ResponseEvaluator:
    """
    Evaluates chatbot responses for quality, accuracy, safety, and completeness
    using multiple models, including OpenAI GPT and local transformer models.
    """
    
    # List of approved sources for reproductive health information
    APPROVED_SOURCES = [
        "planned parenthood", 
        "abortion policy api",
        "american college of obstetricians and gynecologists",
        "acog", 
        "world health organization", 
        "centers for disease control", 
        "cdc",
        "mayo clinic",
        "cleveland clinic",
        "national institutes of health",
        "nih"
    ]
    
    # Sensitive topics requiring special care
    SENSITIVE_TOPICS = [
        "abortion", "pregnancy termination", "miscarriage management",
        "domestic abuse", "sexual assault", "intimate partner violence",
        "sexual dysfunction", "hiv", "aids", "sti", "std",
        "gender identity", "transgender", "sexual orientation"
    ]
    
    # Potentially harmful content patterns
    HARMFUL_PATTERNS = [
        "self-harm", "suicide", "illegal", "diy abortion",
        "coat hanger", "herbal abortion", "homemade abortion",
        "back alley", "self-induced", "harmful",
        "dangerous technique", "illegal procedure"
    ]
    
    def __init__(self, evaluation_model="both"):
        """
        Initialize the response evaluator
        
        Args:
            evaluation_model (str): Model to use for evaluation
                "openai": Use OpenAI's models only
                "local": Use local transformer models only
                "both": Use both (default)
        """
        logger.info(f"Initializing Response Evaluator with model type: {evaluation_model}")
        self.evaluation_model = evaluation_model
        
        try:
            # Set up OpenAI client if using OpenAI models
            if evaluation_model in ["openai", "both"]:
                # Get API key from environment variable
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    logger.warning(
                        "OPENAI_API_KEY not found in environment variables. Using placeholder."
                    )
                    api_key = "placeholder_key"
                    
                self.client = OpenAI(api_key=api_key)
                self.openai_model = "gpt-4o"  # Using GPT-4o for evaluation
                logger.info("OpenAI evaluation model initialized")
            
            # Set up local models if using local models
            if evaluation_model in ["local", "both"]:
                # Load toxicity detection model
                self.toxicity_tokenizer = AutoTokenizer.from_pretrained("martin-ha/toxic-comment-model")
                self.toxicity_model = AutoModelForSequenceClassification.from_pretrained("martin-ha/toxic-comment-model")
                
                # Load text quality evaluation model
                self.quality_tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
                self.quality_model = AutoModelForSequenceClassification.from_pretrained("siebert/sentiment-roberta-large-english")
                
                logger.info("Local evaluation models initialized")
            
            logger.info("Response Evaluator initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Response Evaluator: {str(e)}", exc_info=True)
            raise

    def evaluate_response(self, question: str, response: str, source_info: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate the quality and safety of a response
        
        Args:
            question (str): The user's original question
            response (str): The chatbot's proposed response
            source_info (Dict, optional): Information about the sources used
            
        Returns:
            dict: Evaluation results including scores, issues, and improved response
        """
        try:
            # Perform safety check first
            safety_check = self.perform_safety_check(question, response)
            
            # If response is unsafe, return early with safety issues
            if not safety_check["is_safe"]:
                logger.warning(f"Safety check failed: {safety_check['issues']}")
                return {
                    "score": 0,
                    "issues": safety_check["issues"],
                    "safety_check": safety_check,
                    "improved_response": self.generate_safe_alternative(question, response, safety_check["issues"])
                }
            
            # Check sources if provided
            source_validation = self.validate_sources(source_info) if source_info else {"is_valid": True, "issues": []}
            
            # Evaluate with appropriate model(s)
            if self.evaluation_model == "openai":
                evaluation = self._evaluate_with_openai(question, response, source_validation, safety_check)
            elif self.evaluation_model == "local":
                evaluation = self._evaluate_with_local_models(question, response, source_validation, safety_check)
            else:  # "both"
                openai_eval = self._evaluate_with_openai(question, response, source_validation, safety_check)
                local_eval = self._evaluate_with_local_models(question, response, source_validation, safety_check)
                
                # Combine results from both evaluations
                evaluation = {
                    "score": min(openai_eval.get("score", 5), local_eval.get("score", 5)),
                    "issues": openai_eval.get("issues", []) + local_eval.get("issues", []),
                    "improved_response": openai_eval.get("improved_response", response),
                    "safety_check": safety_check,
                    "source_validation": source_validation,
                    "openai_evaluation": openai_eval,
                    "local_evaluation": local_eval
                }
            
            # Log the evaluation result
            self._log_evaluation(question, response, evaluation)
            
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating response: {str(e)}", exc_info=True)
            # Return a default evaluation that passes the original response
            return {
                "score": 5,
                "issues": ["Evaluation service unavailable"],
                "improved_response": response,
                "safety_check": {"is_safe": True, "issues": []},
                "source_validation": {"is_valid": True, "issues": []}
            }

    def _evaluate_with_openai(self, question, response, source_validation, safety_check):
        """Use OpenAI models to evaluate response quality"""
        try:
            # Include source validation and safety check results in the prompt
            source_issues = "; ".join(source_validation.get("issues", [])) if not source_validation.get("is_valid", True) else "No source issues."
            safety_issues = "; ".join(safety_check.get("issues", [])) if safety_check.get("issues", []) else "No safety issues."
            
            evaluation_prompt = f"""
            You're evaluating the quality of a reproductive health chatbot response. Your task is to assess if the response fully addresses the user's question, provides appropriate detail, and includes proper citations.

            Original question: "{question}"
            
            Proposed response: "{response}"
            
            Source validation: {source_issues}
            Safety check: {safety_issues}
            
            Please evaluate this response on the following criteria:
            1. Comprehensiveness: Does it fully address the question?
            2. Accuracy: Is the information correct based on medical knowledge?
            3. Clarity: Is the response easy to understand?
            4. Level of detail: Does it provide appropriate depth for a health topic?
            5. Emotional tone: Is it empathetic and supportive?
            6. Sources: Does it properly cite reliable sources?
            
            Return a JSON structure with the following fields:
            - "score": 1-10 rating of the overall response quality
            - "issues": List of specific issues or gaps in the response
            - "improved_response": An improved version of the response that addresses any issues
            
            Format the improved response to be comprehensive, clear, and empathetic, with 3-4 paragraphs of appropriate detail. ALWAYS include proper source attribution like "(Source: Planned Parenthood)" at the end of the response.
            """
            
            eval_response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[{
                    "role": "system",
                    "content": "You are an expert evaluator of health information quality and completeness."
                }, {
                    "role": "user",
                    "content": evaluation_prompt
                }],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=1500
            )
            
            # Parse the JSON result
            result_text = eval_response.choices[0].message.content
            try:
                result = json.loads(result_text)
                logger.info(f"OpenAI evaluation completed with score: {result.get('score', 'unknown')}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse OpenAI evaluation result as JSON: {result_text}")
                result = {
                    "score": 5,
                    "issues": ["Error parsing evaluation result"],
                    "improved_response": response
                }
            
            # Add safety and source validation information
            result["safety_check"] = safety_check
            result["source_validation"] = source_validation
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating with OpenAI: {str(e)}", exc_info=True)
            return {
                "score": 5,
                "issues": ["OpenAI evaluation failed"],
                "improved_response": response
            }

    def _evaluate_with_local_models(self, question, response, source_validation, safety_check):
        """Use local transformer models to evaluate response quality"""
        try:
            # Check if response addresses the question effectively
            relevance_score = self._calculate_relevance(question, response)
            
            # Analyze sentiment and quality
            quality_result = self._analyze_quality(response)
            
            # Determine if the response contains meaningful content
            is_substantial = len(response.split()) > 50
            
            # Calculate combined score based on all factors
            base_score = (relevance_score * 5) + (quality_result["positivity"] * 2)
            
            # Adjust score based on other factors
            if not is_substantial:
                base_score -= 2
            if not source_validation.get("is_valid", True):
                base_score -= 3
            
            # Normalize to 1-10 scale
            final_score = max(1, min(10, base_score))
            
            # Identify issues
            issues = []
            if relevance_score < 0.7:
                issues.append("Response does not fully address the question")
            if quality_result["positivity"] < 0.5:
                issues.append("Response could be more positive and supportive")
            if not is_substantial:
                issues.append("Response lacks sufficient detail")
            
            # Add source validation issues
            if source_validation.get("issues"):
                issues.extend(source_validation["issues"])
            
            result = {
                "score": final_score,
                "issues": issues,
                "improved_response": response,  # Local models can't generate improved responses
                "metrics": {
                    "relevance": relevance_score,
                    "positivity": quality_result["positivity"],
                    "is_substantial": is_substantial
                },
                "safety_check": safety_check,
                "source_validation": source_validation
            }
            
            logger.info(f"Local model evaluation completed with score: {final_score}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating with local models: {str(e)}", exc_info=True)
            return {
                "score": 5,
                "issues": ["Local model evaluation failed"],
                "improved_response": response
            }

    def _calculate_relevance(self, question, response):
        """Calculate how well the response addresses the question"""
        try:
            # Simple keyword matching for now
            question_keywords = set(question.lower().split())
            response_lower = response.lower()
            
            matches = sum(1 for keyword in question_keywords if keyword in response_lower and len(keyword) > 3)
            return min(1.0, matches / max(1, len(question_keywords)))
        except Exception as e:
            logger.error(f"Error calculating relevance: {str(e)}", exc_info=True)
            return 0.5

    def _analyze_quality(self, text):
        """Analyze the quality and sentiment of text using local models"""
        try:
            # Tokenize and get sentiment scores
            inputs = self.quality_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.quality_model(**inputs)
            
            # Get positive sentiment score (index 1 is typically positive)
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            positivity = scores[0][1].item()
            
            return {
                "positivity": positivity,
                "raw_scores": scores[0].tolist()
            }
        except Exception as e:
            logger.error(f"Error analyzing quality: {str(e)}", exc_info=True)
            return {"positivity": 0.5, "raw_scores": [0.5, 0.5]}

    def perform_safety_check(self, question: str, response: str) -> Dict[str, Any]:
        """
        Check if the response contains any harmful, unsafe, or misleading content
        
        Args:
            question (str): The user's question
            response (str): The chatbot's response
            
        Returns:
            dict: Safety check results with is_safe flag and issues list
        """
        try:
            issues = []
            
            # Check for harmful patterns
            for pattern in self.HARMFUL_PATTERNS:
                if pattern in question.lower() or pattern in response.lower():
                    issues.append(f"Contains potentially harmful content: '{pattern}'")
            
            # Check for misleading medical advice
            if any(phrase in response.lower() for phrase in ["guaranteed to", "100% effective", "always works", "cures", "miracle"]):
                issues.append("Contains potentially misleading medical claims")
            
            # Check if sensitive topics are addressed carefully
            sensitive_topic_mentioned = any(topic in question.lower() or topic in response.lower() for topic in self.SENSITIVE_TOPICS)
            
            # For sensitive topics, check if tone is appropriate
            if sensitive_topic_mentioned:
                if self.evaluation_model in ["local", "both"]:
                    # Use toxicity model to check response tone
                    toxicity_score = self._check_toxicity(response)
                    if toxicity_score > 0.3:  # Lower threshold for sensitive health topics
                        issues.append(f"Response tone may be inappropriate for sensitive topic (toxicity: {toxicity_score:.2f})")
            
            # If OpenAI is available, use it for comprehensive safety check
            if self.evaluation_model in ["openai", "both"]:
                openai_safety = self._check_safety_with_openai(question, response)
                if openai_safety.get("issues"):
                    issues.extend(openai_safety["issues"])
            
            return {
                "is_safe": len(issues) == 0,
                "issues": issues,
                "sensitive_topic_detected": sensitive_topic_mentioned
            }
            
        except Exception as e:
            logger.error(f"Error performing safety check: {str(e)}", exc_info=True)
            # Default to assuming content is safe when check fails
            return {"is_safe": True, "issues": ["Safety check failed but proceeding with caution"]}

    def _check_toxicity(self, text: str) -> float:
        """Check toxicity level of text using local model"""
        try:
            inputs = self.toxicity_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.toxicity_model(**inputs)
            
            scores = torch.nn.functional.softmax(outputs.logits, dim=1)
            # Typically index 1 is the toxic class
            toxicity_score = scores[0][1].item()
            
            return toxicity_score
        except Exception as e:
            logger.error(f"Error checking toxicity: {str(e)}", exc_info=True)
            return 0.0  # Default to safe

    def _check_safety_with_openai(self, question: str, response: str) -> Dict[str, Any]:
        """Use OpenAI to check for safety issues"""
        try:
            safety_prompt = f"""
            Examine this chatbot response about reproductive health for any potential safety issues:
            
            User question: "{question}"
            Chatbot response: "{response}"
            
            Identify any issues in these categories:
            1. Harmful advice or misinformation
            2. Inappropriate medical recommendations
            3. Misleading pregnancy or abortion information
            4. Stigmatizing or judgmental language
            5. Inaccurate medical facts
            
            Return a JSON structure with:
            - "has_issues": true or false
            - "issues": list of specific safety concerns (empty if none found)
            
            Be particularly alert to any content that could lead to physical or emotional harm.
            """
            
            safety_response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[{
                    "role": "system",
                    "content": "You are a safety expert focusing on reproductive health information."
                }, {
                    "role": "user",
                    "content": safety_prompt
                }],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=500
            )
            
            # Parse the JSON result
            result_text = safety_response.choices[0].message.content
            try:
                result = json.loads(result_text)
                if result.get("has_issues", False):
                    logger.warning(f"OpenAI safety check found issues: {result.get('issues', [])}")
                return result
            except json.JSONDecodeError:
                logger.error(f"Failed to parse OpenAI safety check result as JSON: {result_text}")
                return {"has_issues": False, "issues": []}
            
        except Exception as e:
            logger.error(f"Error checking safety with OpenAI: {str(e)}", exc_info=True)
            return {"has_issues": False, "issues": []}

    def validate_sources(self, source_info: Optional[Dict]) -> Dict[str, Any]:
        """
        Check if the sources used are approved and properly cited
        
        Args:
            source_info (Dict): Information about the sources used
            
        Returns:
            dict: Validation results with is_valid flag and issues list
        """
        if not source_info:
            return {"is_valid": False, "issues": ["No source information provided"]}
        
        try:
            issues = []
            sources_mentioned = []
            
            # Extract sources from the information
            if "citations" in source_info:
                sources_mentioned = [citation.get("source", "").lower() for citation in source_info["citations"]]
            elif "text" in source_info:
                # Try to extract source mentions from text
                text_lower = source_info["text"].lower()
                sources_mentioned = [source for source in self.APPROVED_SOURCES if source in text_lower]
            
            # Check if any sources are mentioned
            if not sources_mentioned:
                issues.append("No recognized sources cited")
            
            # Check if sources are approved
            unapproved_sources = []
            for source in sources_mentioned:
                if not any(approved in source for approved in self.APPROVED_SOURCES):
                    unapproved_sources.append(source)
            
            if unapproved_sources:
                issues.append(f"Unapproved sources cited: {', '.join(unapproved_sources)}")
            
            return {
                "is_valid": len(issues) == 0,
                "issues": issues,
                "sources_detected": sources_mentioned
            }
            
        except Exception as e:
            logger.error(f"Error validating sources: {str(e)}", exc_info=True)
            return {"is_valid": True, "issues": ["Source validation failed but proceeding with caution"]}

    def generate_safe_alternative(self, question: str, response: str, issues: List) -> str:
        """
        Generate a safe alternative response when safety issues are detected
        
        Args:
            question (str): Original question
            response (str): Problematic response
            issues (List): List of safety issues (can be strings or dicts)
            
        Returns:
            str: A safer alternative response
        """
        try:
            if self.evaluation_model in ["openai", "both"]:
                # Process issues to ensure they're all strings
                processed_issues = []
                for issue in issues:
                    if isinstance(issue, dict):
                        # Extract relevant information from dictionary
                        if 'category' in issue and 'concern' in issue:
                            processed_issues.append(f"{issue['category']}: {issue['concern']}")
                        else:
                            # Convert dictionary to JSON string
                            processed_issues.append(json.dumps(issue))
                    else:
                        # Convert any non-string issues to strings
                        processed_issues.append(str(issue))
                
                issues_text = "; ".join(processed_issues)
                
                safety_prompt = f"""
                The following chatbot response to a reproductive health question has safety issues:
                
                User question: "{question}"
                Problematic response: "{response}"
                
                Safety issues: {issues_text}
                
                Create a safer alternative response that:
                1. Addresses the user's question if appropriate
                2. Avoids ALL the safety issues listed
                3. Uses medically accurate, non-judgmental language
                4. If the question itself requests harmful information, gently redirect
                5. Includes proper source attribution (Planned Parenthood or similar)
                
                Your response should be informative but prioritize user safety above all.
                """
                
                safe_response = self.client.chat.completions.create(
                    model=self.openai_model,
                    messages=[{
                        "role": "system",
                        "content": "You are an expert in reproductive health safety and ethics."
                    }, {
                        "role": "user",
                        "content": safety_prompt
                    }],
                    temperature=0.3,
                    max_tokens=800
                )
                
                return safe_response.choices[0].message.content
            else:
                # Fallback generic safe response when OpenAI isn't available
                return "I want to provide you with accurate information on this topic. Let me connect you with reliable resources from Planned Parenthood or healthcare providers who can give you the best guidance. Is there a specific aspect of reproductive health I can help you understand better with factual information? (Source: Planned Parenthood)"
                
        except Exception as e:
            logger.error(f"Error generating safe alternative: {str(e)}", exc_info=True)
            return "I apologize, but I need to provide only medically accurate information from trusted sources. Could you rephrase your question, or may I help you with a different reproductive health topic? (Source: Planned Parenthood)"

    def should_use_improved_response(self, evaluation_result: Dict[str, Any]) -> bool:
        """
        Determine if the improved response should be used
        
        Args:
            evaluation_result (dict): The evaluation result from evaluate_response
            
        Returns:
            bool: True if the improved response should be used
        """
        try:
            # Always use improved response if safety issues were found
            if "safety_check" in evaluation_result and not evaluation_result["safety_check"].get("is_safe", True):
                logger.info("Using improved response due to safety concerns")
                return True
            
            # Always use improved response if source issues were found
            if "source_validation" in evaluation_result and not evaluation_result["source_validation"].get("is_valid", True):
                logger.info("Using improved response due to source validation concerns")
                return True
            
            # Use the improved response if the original score is below 7
            score = evaluation_result.get('score', 5)
            return score < 7
        except Exception as e:
            logger.error(f"Error determining if improved response should be used: {str(e)}", exc_info=True)
            return False
            
    def get_improved_response(self, question: str, response: str, source_info: Optional[Dict] = None) -> str:
        """
        Evaluate and potentially improve a response
        
        Args:
            question (str): The user's original question
            response (str): The chatbot's proposed response
            source_info (Dict, optional): Information about the sources used
            
        Returns:
            str: Either the original or an improved response
        """
        try:
            # Skip evaluation for very short responses (likely conversational)
            if len(response.split()) < 20:
                return response
                
            # Get the evaluation result
            evaluation_result = self.evaluate_response(question, response, source_info)
            
            # Determine if the improved response should be used
            if self.should_use_improved_response(evaluation_result):
                logger.info("Using improved response based on evaluation")
                improved = evaluation_result.get('improved_response', response)
                return improved
            else:
                logger.info("Using original response based on evaluation")
                return response
                
        except Exception as e:
            logger.error(f"Error in get_improved_response: {str(e)}", exc_info=True)
            return response  # Fall back to original response
    
    def _log_evaluation(self, question: str, response: str, evaluation: Dict[str, Any]) -> None:
        """
        Log evaluation results to file for analysis and record metrics
        
        Args:
            question (str): The original question
            response (str): The original response
            evaluation (Dict): The evaluation results
        """
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "response": response,
                "evaluation": evaluation,
            }
            
            # Write to the JSON log file
            with open('evaluation_logs.json', 'a') as f:
                f.write(json.dumps(log_entry) + "\n")
            
            # Record metrics for AWS CloudWatch
            score = evaluation.get("score", 0)
            
            # Track overall evaluation counts
            increment_counter("evaluations_total")
            
            # Track if response was improved
            if evaluation.get("improved_response", "") != response:
                increment_counter("evaluations_improved")
            
            # Track safety issues
            safety = evaluation.get("safety_check", {})
            if safety and not safety.get("is_safe", True):
                increment_counter("safety_issues")
                
                # Track specific types of safety issues
                issues = safety.get("issues", [])
                for issue in issues:
                    if isinstance(issue, dict) and "type" in issue:
                        increment_counter(f"safety_issue_{issue['type']}")
                    elif isinstance(issue, str):
                        # Extract a category from the issue string
                        if "harmful" in issue.lower():
                            increment_counter("safety_issue_harmful")
                        elif "misleading" in issue.lower():
                            increment_counter("safety_issue_misleading")
                        else:
                            increment_counter("safety_issue_other")
            
            # Track quality scores in ranges
            if score <= 3:
                increment_counter("quality_score_low")
            elif score <= 6:
                increment_counter("quality_score_medium")
            else:
                increment_counter("quality_score_high")
                
            # Track OpenAI token usage if this was an OpenAI evaluation
            if self.evaluation_model in ["openai", "both"]:
                # Rough estimate of tokens used in evaluation
                approx_tokens = (len(question.split()) + len(response.split())) * 2
                record_api_call("openai_evaluation", approx_tokens)
                
        except Exception as e:
            logger.error(f"Error logging evaluation: {str(e)}", exc_info=True)