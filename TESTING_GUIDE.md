# Reproductive Health Chatbot Testing Framework Guide

This guide provides instructions for using the comprehensive testing framework for the reproductive health chatbot. The framework includes tools for testing metrics tracking, citation management, topic detection, and response quality.

## Overview

The testing framework consists of multiple components:

1. **Metrics Tracking Tests**: Verify the collection and tracking of performance metrics, including inference time by category, response quality, and real-time ROUGE metrics.

2. **Citation System Tests**: Verify that appropriate external source citations are added to responses, especially when RAG data is insufficient.

3. **Topic Detection Tests**: Verify the accurate identification of reproductive health topics and question categories.

4. **ROUGE Metrics Tests**: Measure and track text similarity metrics for response quality evaluation.

5. **Comprehensive Test Runner**: Executes all tests and compiles results into a detailed report.

## Prerequisites

Make sure all required packages are installed:

```bash
pip install -r requirements.txt
```

## Running Individual Tests

### Metrics Tracking Tests

Tests the metrics tracking functionality by simulating questions across different categories and topics:

```bash
python scripts/test_metrics_tracking.py
```

Output will be saved to `metrics_test_results.json`.

### ROUGE Metrics Tests

Tests the ROUGE metrics calculation and tracking functionality:

```bash
python scripts/test_rouge_metrics.py
```

Output will be saved to `rouge_metrics_test_results.json`.

### Citation System Tests

Tests the citation management system, including addition of trusted sources when RAG data is insufficient:

```bash
python scripts/test_citation_system.py
```

Output will be saved to `citation_system_test_results.json`.

### Topic Detection Tests

Tests reproductive health topic detection and question categorization accuracy:

```bash
python scripts/test_topic_detection.py
```

Output will be saved to `topic_detection_test_results.json`.

## Running Comprehensive Tests

To run all tests and generate a comprehensive report, use the main test runner:

```bash
python run_comprehensive_tests.py
```

This will execute all test modules sequentially and compile the results into a comprehensive report file named `comprehensive_test_report_YYYYMMDD_HHMMSS.json`.

## Test Result Analysis

Each test produces a detailed JSON output file containing test results. The comprehensive report aggregates these results and includes summary statistics including:

- Overall topic detection accuracy
- Real-time ROUGE metrics
- Category-based inference time statistics
- Citation system effectiveness
- Response quality metrics

## Interpreting Metrics

### Inference Time Metrics

- **Average Inference Time**: Overall average time taken to generate a response
- **Category-specific Inference Time**: Average time broken down by question category (policy, knowledge, conversational, etc.)
- **Topic-specific Inference Time**: Average time broken down by reproductive health topic

### ROUGE Metrics

- **ROUGE-1**: Measures unigram overlap between the generated and reference texts
- **ROUGE-2**: Measures bigram overlap between the generated and reference texts
- **ROUGE-L**: Measures the longest common subsequence between the generated and reference texts

Higher ROUGE scores (0-1 scale) indicate better similarity between generated responses and reference texts.

### Citation Metrics

- **Citation Count**: Number of citations included in responses
- **Authoritative Source Rate**: Percentage of responses that include citations from authoritative healthcare sources
- **Source Distribution**: Breakdown of citation sources by organization

## Performance Tuning

Based on test results, you may want to adjust the chatbot's performance in various ways:

1. **For slow inference times on specific topics**: Optimize the retrieval process for those topics, potentially using pre-computed embeddings or caching.

2. **For low ROUGE scores**: Fine-tune the response generation process, potentially adding more context to the prompt or improving the quality of the RAG corpus.

3. **For citation issues**: Update the trusted sources list or fine-tune the threshold for when to add external citations.

## Dashboard Integration

Test results can be viewed in the admin dashboard, which includes visualizations of:

- Category-specific inference times
- Real-time ROUGE metrics
- Topic detection accuracy
- Citation effectiveness

To access the dashboard, navigate to `/admin/dashboard` in your browser when the application is running.

## Adding New Tests

To add new tests to the framework:

1. Create a new test script in the `scripts/` directory
2. Update `TEST_MODULES` and `RESULT_FILES` in `run_comprehensive_tests.py`
3. Add any necessary visualization components to the dashboard

## Troubleshooting

If you encounter issues with the testing framework:

- Check for missing dependencies
- Verify that the database is properly initialized
- Ensure all model files are present
- Check that the OpenAI API key is set if using GPT for question classification or response enhancement

## Best Practices

- Run comprehensive tests after any significant changes to the chatbot
- Monitor real-time metrics during production usage
- Regularly analyze test results to identify areas for improvement
- Keep reference datasets updated with latest reproductive health information