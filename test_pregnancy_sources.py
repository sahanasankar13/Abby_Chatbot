#!/usr/bin/env python3
"""
Test script to directly test pregnancy planning detection and trusted sources functions.
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

def test_pregnancy_planning_functions():
    """Test the pregnancy planning detection and source functions directly"""
    logger.info("Testing pregnancy planning functions directly...")
    
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
            
        def _is_pregnancy_planning_question(self, question):
            """Directly copied from BaselineModel"""
            question_lower = question.lower()
            planning_keywords = [
                "trying to conceive", "trying to get pregnant", "ttc", 
                "fertility", "ovulation", "conception", "conceive",
                "getting pregnant", "preconception", "pre-conception",
                "prepare for pregnancy", "planning for pregnancy", 
                "before getting pregnant", "before pregnancy"
            ]
            
            return any(keyword in question_lower for keyword in planning_keywords)
            
        def _add_trusted_pregnancy_sources(self, response, question):
            """Simplified version of the method in BaselineModel"""
            source_info = {
                "source": "planned_parenthood",
                "citations": [{"source": "Planned Parenthood", "url": "https://www.plannedparenthood.org/"}]
            }
            
            # Add a section with additional sources
            additional_sources = "\n\nFor more detailed information on pregnancy planning, you may want to check these trusted resources:\n"
            
            # Analyze question to determine relevant sources
            question_lower = question.lower()
            
            if "preconception" in question_lower or "prepare" in question_lower or "planning" in question_lower:
                additional_sources += "- Mayo Clinic: Preconception planning (Source: Mayo Clinic)\n"
                source_info["citations"].append({"source": "Mayo Clinic", "url": "https://www.mayoclinic.org/healthy-lifestyle/getting-pregnant/in-depth/preconception-planning/art-20047296"})
            
            if "fertility" in question_lower or "infertility" in question_lower:
                additional_sources += "- National Institutes of Health: Fertility and Infertility (Source: National Institutes of Health)\n"
                source_info["citations"].append({"source": "National Institutes of Health", "url": "https://www.nichd.nih.gov/health/topics/fertility"})
            
            # Always add CDC and ACOG as general resources
            additional_sources += "- Centers for Disease Control and Prevention: Planning for Pregnancy (Source: Centers for Disease Control and Prevention)\n"
            source_info["citations"].append({"source": "Centers for Disease Control and Prevention", "url": "https://www.cdc.gov/preconception/planning.html"})
            
            additional_sources += "- American College of Obstetricians and Gynecologists: Pregnancy Resources (Source: American College of Obstetricians and Gynecologists)"
            source_info["citations"].append({"source": "American College of Obstetricians and Gynecologists", "url": "https://www.acog.org/womens-health/pregnancy"})
            
            enhanced_response = response + additional_sources
            return enhanced_response, source_info
    
    # Create a mock instance
    mock_model = MockBaselineModel()
    logger.info("Mock model initialized successfully")
    
    # Test questions focused on pregnancy planning
    test_questions = [
        "What should I do when trying to get pregnant?",
        "How can I increase my fertility?",
        "What are the best ways to prepare for pregnancy?",
        "What nutritional changes should I make before conceiving?",
        "How can I track ovulation for conception?",
        # Add some non-pregnancy planning questions as control
        "What is the law on abortion in Texas?",
        "How does birth control work?",
        "What are the symptoms of a UTI?"
    ]
    
    logger.info(f"Testing {len(test_questions)} questions")
    
    for i, question in enumerate(test_questions):
        logger.info(f"Testing question {i+1}: {question}")
        
        # Check if it's correctly detected as a pregnancy planning question
        is_pregnancy_planning = mock_model._is_pregnancy_planning_question(question)
        logger.info(f"Detected as pregnancy planning question: {is_pregnancy_planning}")
        
        # Only add sources if it's a pregnancy planning question
        if is_pregnancy_planning:
            # Create a sample RAG response
            sample_response = f"Here is some information about {question}: This is a simulated RAG response."
            
            # Add trusted sources to the response
            enhanced_response, source_info = mock_model._add_trusted_pregnancy_sources(sample_response, question)
            
            # Check if the response includes trusted sources
            includes_sources = any(source in enhanced_response for source in [
                "Mayo Clinic", 
                "National Institutes of Health",
                "Centers for Disease Control and Prevention",
                "American College of Obstetricians and Gynecologists"
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
        success = test_pregnancy_planning_functions()
        if success:
            logger.info("All tests completed successfully")
        else:
            logger.error("Tests failed")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Error during testing: {e}", exc_info=True)
        sys.exit(1)