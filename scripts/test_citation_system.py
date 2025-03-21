"""
Test Citation System

This script tests the citation management system of the reproductive health chatbot,
verifying that appropriate external source citations are added to responses and
that the citation format meets quality standards.
"""

import sys
import os
import json
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot.citation_manager import CitationManager, Citation
from chatbot.baseline_model import BaselineModel
from chatbot.bert_rag import BertRAGModel
from utils.metrics import get_metrics, flush_metrics

def test_citation_manager():
    """Test basic citation manager functionality"""
    print("\n=== Testing Citation Manager ===")
    
    # Initialize the citation manager
    citation_manager = CitationManager()
    
    # Test adding citations to text
    test_text = "Birth control pills help prevent pregnancy."
    
    # Test with Planned Parenthood source
    pp_text = citation_manager.add_citation_to_text(test_text, "planned_parenthood")
    print(f"Original: {test_text}")
    print(f"With PP citation: {pp_text}")
    
    # Test with ACOG source
    acog_text = citation_manager.add_citation_to_text(test_text, "acog")
    print(f"With ACOG citation: {acog_text}")
    
    # Test extracting citations
    citations = citation_manager.extract_citations_from_text(pp_text)
    print(f"\nExtracted citations: {len(citations)}")
    for citation in citations:
        print(f"- {citation.source} ({citation.url})")
    
    # Test formatting with different formats
    formatted_html = citation_manager.format_response_with_citations(pp_text, "html")
    formatted_md = citation_manager.format_response_with_citations(pp_text, "markdown")
    
    print(f"\nHTML formatted:\n{formatted_html['text']}")
    print(f"Citations: {len(formatted_html['citations'])}")
    
    print(f"\nMarkdown formatted:\n{formatted_md['text']}")
    print(f"Citations: {len(formatted_md['citations'])}")


def test_trusted_sources_addition():
    """Test the addition of trusted sources to responses based on topic"""
    print("\n=== Testing Trusted Sources Addition ===")
    
    # Initialize the model
    model = BaselineModel()
    
    # Test questions by reproductive health topics
    topics = {
        "pregnancy": "What are the signs of pregnancy?",
        "birth_control": "How effective is the pill?",
        "abortion": "What is a medication abortion?",
        "menstruation": "Why is my period irregular?",
        "pregnancy_planning": "How can I increase my chances of conception?"
    }
    
    for topic, question in topics.items():
        print(f"\nTesting topic: {topic}")
        print(f"Question: {question}")
        
        # Get the response
        response = model.process_question(question, force_category="knowledge")
        
        # Check for citations
        citations = CitationManager().extract_citations_from_text(response)
        
        print(f"Response length: {len(response)} chars")
        print(f"Citations found: {len(citations)}")
        
        for citation in citations:
            print(f"- {citation.source} ({citation.url})")


def test_sources_when_rag_insufficient():
    """Test the addition of trusted sources when RAG data is insufficient"""
    print("\n=== Testing Sources When RAG Data is Insufficient ===")
    
    # Initialize the model components
    model = BaselineModel()
    bert_rag = BertRAGModel()
    
    # Test with a question that likely isn't well-covered in the RAG data
    edge_questions = [
        "What are the latest recommendations for HPV vaccination?",
        "How does endometriosis affect fertility?",
        "What are the risks of untreated chlamydia in men?",
        "How effective are fertility tracking apps?",
        "What are the mental health impacts of pregnancy loss?"
    ]
    
    for question in edge_questions:
        print(f"\nQuestion: {question}")
        
        # Check if RAG is confident
        rag_response, contexts = bert_rag.get_response_with_context(question)
        is_confident = bert_rag.is_confident(question, rag_response)
        
        print(f"RAG confident: {is_confident}")
        
        # Get the full model response
        full_response = model.process_question(question, force_category="knowledge")
        
        # Check for citations
        citations = CitationManager().extract_citations_from_text(full_response)
        
        print(f"Citations found: {len(citations)}")
        for citation in citations:
            print(f"- {citation.source}")
        
        # Verify response length
        if not is_confident:
            print(f"RAG not confident, full response length: {len(full_response)} chars")
            if len(full_response) < len(rag_response):
                print("WARNING: Full response is shorter than RAG response")


def check_authoritative_sources(citations: List[Citation]) -> bool:
    """Check if citations include authoritative healthcare sources"""
    authoritative_sources = [
        "planned parenthood",
        "american college of obstetricians and gynecologists",
        "acog",
        "centers for disease control",
        "cdc",
        "world health organization",
        "who",
        "mayo clinic",
        "cleveland clinic",
        "national institutes of health",
        "nih"
    ]
    
    for citation in citations:
        source_lower = citation.source.lower()
        if any(auth_source in source_lower for auth_source in authoritative_sources):
            return True
    return False


def test_comprehensive_citation():
    """Run a comprehensive test of the citation system"""
    print("\n=== Running Comprehensive Citation System Test ===")
    
    # Initialize the model
    model = BaselineModel()
    
    # Test with a diverse set of questions
    test_questions = [
        # Policy questions
        {"question": "Is abortion legal in Florida?", "expected_category": "policy"},
        {"question": "What are Texas abortion laws?", "expected_category": "policy"},
        
        # Medical knowledge questions
        {"question": "What are the symptoms of pregnancy?", "expected_category": "knowledge"},
        {"question": "How does an IUD work?", "expected_category": "knowledge"},
        
        # Mixed questions
        {"question": "What are the different types of abortion and what states allow them?", "expected_category": "knowledge"},
        
        # Edge questions
        {"question": "What are the psychological effects of abortion?", "expected_category": "knowledge"},
        {"question": "Are there any new birth control methods being developed?", "expected_category": "knowledge"}
    ]
    
    for test_case in test_questions:
        question = test_case["question"]
        print(f"\nQuestion: {question}")
        
        # Get the response
        response = model.process_question(question)
        
        # Extract citations
        citations = CitationManager().extract_citations_from_text(response)
        
        # Check for authoritative sources
        has_authoritative = check_authoritative_sources(citations)
        
        print(f"Response length: {len(response)} chars")
        print(f"Citations found: {len(citations)}")
        print(f"Has authoritative sources: {has_authoritative}")
        
        # List the sources
        for citation in citations:
            print(f"- {citation.source}")
        
        # Basic verification
        if len(citations) == 0:
            print("WARNING: No citations found in response")
        
        # Print snippet of response
        snippet = response[:200] + "..." if len(response) > 200 else response
        print(f"\nResponse snippet: {snippet}")


def run_tests():
    """Run all citation system tests"""
    # Reset metrics
    flush_metrics(reset=True)
    
    # Run the tests
    test_citation_manager()
    test_trusted_sources_addition()
    test_sources_when_rag_insufficient()
    test_comprehensive_citation()
    
    # Save the results summary
    results = {
        "tests_run": [
            "citation_manager",
            "trusted_sources_addition",
            "sources_when_rag_insufficient",
            "comprehensive_citation"
        ],
        "metrics": get_metrics()
    }
    
    with open("citation_system_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nTest results saved to citation_system_test_results.json")


if __name__ == "__main__":
    run_tests()