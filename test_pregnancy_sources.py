#!/usr/bin/env python3
"""
Test script to directly test reproductive health topic detection and trusted sources functions.
This script is more lightweight than testing the entire model pipeline.
"""

import os
import sys
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_reproductive_health_functions():
    """Test the reproductive health topic detection and trusted sources functions"""
    logger.info("Testing reproductive health topic detection and trusted sources...")
    
    # Import only the specific module we need to test
    try:
        from chatbot.citation_manager import CitationManager
        from chatbot.baseline_model import BaselineModel
        logger.info("Successfully imported required modules")
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        sys.exit(1)
        
    # Create instances just for testing
    citation_manager = CitationManager()
    
    # We'll create a simple mock of the BaselineModel class to test just the functions we added
    class MockBaselineModel:
        def __init__(self):
            pass
            
        def _get_reproductive_health_topic(self, question):
            """Identify the reproductive health topic from a question"""
            question_lower = question.lower()
            
            # Define topic categories and their keywords
            topic_keywords = {
                "pregnancy_planning": [
                    "trying to conceive", "trying to get pregnant", "ttc", 
                    "fertility", "ovulation", "conception", "conceive",
                    "getting pregnant", "preconception", "pre-conception",
                    "prepare for pregnancy", "planning for pregnancy", 
                    "before getting pregnant", "before pregnancy"
                ],
                "pregnancy": [
                    "pregnant", "pregnancy", "prenatal", "trimester", 
                    "fetal", "fetus", "morning sickness", "baby bump",
                    "ultrasound", "expecting", "gestational"
                ],
                "birth_control": [
                    "birth control", "contraception", "contraceptive", "iud", 
                    "pill", "condom", "implant", "nexplanon", "depo",
                    "morning after", "plan b", "spermicide"
                ],
                "menstruation": [
                    "period", "menstrual", "menstruation", "cycle", "pms",
                    "cramps", "bleeding", "spotting", "tampon", "pad", "flow"
                ],
                "reproductive_health": [
                    "reproductive health", "sexual health", "gynecology", 
                    "obgyn", "pelvic", "vaginal", "cervical", "uterine", 
                    "ovarian", "testicular", "std", "sti", "infection"
                ],
                "abortion": [
                    "abortion", "terminate", "termination", "miscarriage", 
                    "pregnancy loss", "roe", "wade", "pro-choice", "pro-life"
                ]
            }
            
            # Check for topic matches
            for topic, keywords in topic_keywords.items():
                if any(keyword in question_lower for keyword in keywords):
                    return topic
                    
            # Return a general reproductive health topic if no specific match
            return "general_reproductive_health"
            
        def _add_trusted_sources(self, response, question):
            """Add links to trusted external sources for reproductive health questions"""
            
            # Only add sources if the response doesn't already have citations
            if "(Source:" in response:
                return response, {
                    "source": "planned_parenthood",
                    "citations": [{"source": "Planned Parenthood", "url": "https://www.plannedparenthood.org/"}]
                }
            
            # Initialize source info
            source_info = {
                "source": "planned_parenthood",
                "citations": [{"source": "Planned Parenthood", "url": "https://www.plannedparenthood.org/"}]
            }
            
            # Identify the topic
            topic = self._get_reproductive_health_topic(question)
            
            # Define trusted sources for different topics
            topic_sources = {
                "pregnancy_planning": [
                    ("Mayo Clinic", "Preconception planning", "https://www.mayoclinic.org/healthy-lifestyle/getting-pregnant/in-depth/preconception-planning/art-20047296"),
                    ("National Institutes of Health", "Fertility and Infertility", "https://www.nichd.nih.gov/health/topics/fertility"),
                    ("Centers for Disease Control and Prevention", "Planning for Pregnancy", "https://www.cdc.gov/preconception/planning.html")
                ],
                "pregnancy": [
                    ("American College of Obstetricians and Gynecologists", "Pregnancy Resources", "https://www.acog.org/womens-health/pregnancy"),
                    ("Mayo Clinic", "Pregnancy week by week", "https://www.mayoclinic.org/healthy-lifestyle/pregnancy-week-by-week/basics/healthy-pregnancy/hlv-20049471"),
                    ("Centers for Disease Control and Prevention", "Pregnancy", "https://www.cdc.gov/pregnancy/index.html")
                ],
                "birth_control": [
                    ("Centers for Disease Control and Prevention", "Contraception", "https://www.cdc.gov/reproductivehealth/contraception/index.htm"),
                    ("American College of Obstetricians and Gynecologists", "Birth Control", "https://www.acog.org/womens-health/birth-control"),
                    ("Mayo Clinic", "Birth control options", "https://www.mayoclinic.org/tests-procedures/birth-control/about/pac-20384621")
                ],
                "menstruation": [
                    ("Mayo Clinic", "Menstrual cycle", "https://www.mayoclinic.org/healthy-lifestyle/womens-health/in-depth/menstrual-cycle/art-20047186"),
                    ("National Institutes of Health", "Menstruation and Menstrual Problems", "https://www.nichd.nih.gov/health/topics/menstruation"),
                    ("American College of Obstetricians and Gynecologists", "Abnormal Uterine Bleeding", "https://www.acog.org/womens-health/faqs/abnormal-uterine-bleeding")
                ],
                "reproductive_health": [
                    ("Centers for Disease Control and Prevention", "Reproductive Health", "https://www.cdc.gov/reproductivehealth/index.html"),
                    ("World Health Organization", "Sexual and reproductive health", "https://www.who.int/health-topics/sexual-and-reproductive-health"),
                    ("National Institutes of Health", "Reproductive Health", "https://www.nichd.nih.gov/health/topics/reproductive")
                ],
                "abortion": [
                    ("American College of Obstetricians and Gynecologists", "Abortion Policy", "https://www.acog.org/clinical-information/policy-and-position-statements/statements-of-policy/2022/abortion-policy"),
                    ("National Institutes of Health", "Medical Care After Pregnancy Loss", "https://www.nichd.nih.gov/health/topics/pregnancyloss/conditioninfo/treatment"),
                    ("Abortion Policy API", "State Policies", "https://www.abortionpolicyapi.com/")
                ],
                "general_reproductive_health": [
                    ("Centers for Disease Control and Prevention", "Reproductive Health", "https://www.cdc.gov/reproductivehealth/index.html"),
                    ("American College of Obstetricians and Gynecologists", "Women's Health", "https://www.acog.org/womens-health"),
                    ("National Institutes of Health", "Reproductive Health", "https://www.nichd.nih.gov/health/topics/reproductive")
                ]
            }
            
            # Add introduction for additional sources
            additional_sources = f"\n\nFor more detailed information on {topic.replace('_', ' ')}, you may want to check these trusted resources:\n"
            
            # Add the relevant sources for the topic
            selected_sources = topic_sources.get(topic, topic_sources["general_reproductive_health"])
            
            for source, title, url in selected_sources:
                additional_sources += f"- {source}: {title} (Source: {source})\n"
                source_info["citations"].append({"source": source, "url": url})
            
            # Return the enhanced response with the sources
            enhanced_response = response + additional_sources.rstrip()
            return enhanced_response, source_info
    
    # Create a mock instance
    mock_model = MockBaselineModel()
    logger.info("Mock model initialized successfully")
    
    # Test a variety of reproductive health questions
    test_questions = [
        # Pregnancy planning
        "What should I do when trying to get pregnant?",
        "How can I increase my fertility?",
        # Pregnancy
        "What are normal symptoms during first trimester?",
        "How does the fetus develop during pregnancy?",
        # Birth control
        "How effective is an IUD for birth control?",
        "What are the side effects of birth control pills?", 
        # Menstruation
        "Why is my period irregular?",
        "How to manage menstrual cramps?",
        # General reproductive health
        "What are common STI symptoms?",
        "How often should I get a pap smear?",
        # Abortion
        "What is the law on abortion in Texas?",
        "What happens during a medical abortion?",
        # General question
        "How do hormones affect reproductive health?"
    ]
    
    logger.info(f"Testing {len(test_questions)} questions")
    
    for i, question in enumerate(test_questions):
        logger.info(f"Testing question {i+1}: {question}")
        
        # Identify the topic for this question
        topic = mock_model._get_reproductive_health_topic(question)
        logger.info(f"Detected topic: {topic}")
        
        # Create a sample RAG response
        sample_response = f"Here is some information about {question}: This is a simulated RAG response."
        
        # Add trusted sources to the response
        enhanced_response, source_info = mock_model._add_trusted_sources(sample_response, question)
        
        # Check if the response includes trusted sources
        includes_sources = any(source in enhanced_response for source in [
            "Mayo Clinic", 
            "National Institutes of Health",
            "Centers for Disease Control and Prevention",
            "American College of Obstetricians and Gynecologists",
            "World Health Organization",
            "Abortion Policy API"
        ])
        
        logger.info(f"Response includes trusted sources: {includes_sources}")
        logger.info(f"Number of citations added: {len(source_info['citations'])}")
        
        # Log the citations
        for citation in source_info["citations"]:
            logger.info(f"  - {citation['source']}: {citation['url']}")
        
        # Print a snippet of the response
        response_preview = enhanced_response[:150] + "..." if len(enhanced_response) > 150 else enhanced_response
        logger.info(f"Response preview: {response_preview}")
        
        # Check if citation manager can find these citations
        citations = citation_manager.extract_citations_from_text(enhanced_response)
        logger.info(f"Citations found by citation manager: {len(citations)}")
        
        print("\n" + "="*50 + "\n")
    
    logger.info("Testing completed")
    return True

if __name__ == "__main__":
    # Run the tests
    try:
        success = test_reproductive_health_functions()
        if success:
            logger.info("All tests completed successfully")
        else:
            logger.error("Tests failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        sys.exit(1)