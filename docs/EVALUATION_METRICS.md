# Evaluation Metrics for Reproductive Health Chatbot

This document describes the comprehensive evaluation metrics used to assess the performance and quality of the reproductive health chatbot.

## Real-Time Performance Metrics

### Inference Time

Measures the time taken to generate a response, categorized by question type:

| Metric | Description |
|--------|-------------|
| `average_inference_time_ms` | Average time (in ms) taken to generate a response |
| `inference_time_by_category` | Average time broken down by question category (policy, knowledge, conversational, etc.) |
| `inference_time_by_topic` | Average time broken down by reproductive health topic (pregnancy, birth control, abortion, etc.) |

### Memory Usage

Tracks memory consumption during response generation:

| Metric | Description |
|--------|-------------|
| `average_memory_usage_mb` | Average memory usage in MB per response |
| `max_memory_usage_mb` | Maximum memory usage observed |

### Token Usage

Monitors token consumption for LLM-based operations:

| Metric | Description |
|--------|-------------|
| `average_tokens_per_response` | Average number of tokens in generated responses |
| `tokens_per_category` | Token usage broken down by question category |
| `api_tokens_used` | Counts of tokens used in API calls (for OpenAI models) |

## Response Quality Metrics

### ROUGE Metrics

Text similarity metrics calculated in real-time:

| Metric | Description | Range |
|--------|-------------|-------|
| `rouge1` | Unigram overlap between generated and reference texts | 0-1 |
| `rouge2` | Bigram overlap between generated and reference texts | 0-1 |
| `rougeL` | Longest common subsequence between generated and reference texts | 0-1 |

### BLEU Score

Measures the quality of text translated from one language to another, used here to evaluate response quality against a reference corpus:

| Metric | Description | Range |
|--------|-------------|-------|
| `bleu_score` | Average BLEU score for responses compared to reference texts | 0-100 |

### BERTScore

Uses contextual embeddings to evaluate semantic similarity:

| Metric | Description | Range |
|--------|-------------|-------|
| `bert_score_precision` | Precision-oriented BERTScore | 0-1 |
| `bert_score_recall` | Recall-oriented BERTScore | 0-1 |
| `bert_score_f1` | F1-oriented BERTScore | 0-1 |

## RAG Evaluation Metrics (Ragas)

Metrics specific to Retrieval-Augmented Generation:

| Metric | Description | Range |
|--------|-------------|-------|
| `faithfulness` | Measures how faithful the generated answer is to the retrieved context | 0-1 |
| `context_precision` | Measures how precisely the retrieved context matches the question | 0-1 |
| `context_recall` | Measures how well the retrieved context covers information needed for the answer | 0-1 |

## Retrieval Accuracy Metrics

Evaluates the effectiveness of the retrieval mechanism:

| Metric | Description | Range |
|--------|-------------|-------|
| `precision@k` | Proportion of relevant documents among top K retrieved documents | 0-1 |
| `recall@k` | Proportion of relevant documents retrieved among all relevant documents | 0-1 |
| `mrr` | Mean Reciprocal Rank - average of the reciprocal ranks of the first relevant item | 0-1 |

## Citation and Source Metrics

Tracks citation usage and source validity:

| Metric | Description |
|--------|-------------|
| `citations_per_response` | Average number of citations included in responses |
| `authoritative_source_rate` | Percentage of responses that include authoritative healthcare sources |
| `source_distribution` | Distribution of citation sources by organization |

## Reproductive Health Topic Detection

Evaluates the accuracy of topic detection:

| Metric | Description |
|--------|-------------|
| `topic_detection_accuracy` | Overall accuracy of reproductive health topic detection |
| `topic_wise_accuracy` | Accuracy broken down by reproductive health topic |

## Question Categorization

Evaluates the accuracy of question classification:

| Metric | Description |
|--------|-------------|
| `category_accuracy` | Overall accuracy of question categorization |
| `category_wise_accuracy` | Accuracy broken down by question category |

## User Feedback Metrics

Tracks user satisfaction with the responses:

| Metric | Description |
|--------|-------------|
| `positive_feedback_rate` | Percentage of responses with positive user feedback |
| `negative_feedback_rate` | Percentage of responses with negative user feedback |
| `feedback_by_topic` | Feedback statistics broken down by reproductive health topic |
| `feedback_by_category` | Feedback statistics broken down by question category |

## Response Improvement Metrics

Tracks the effectiveness of response evaluation and improvement:

| Metric | Description |
|--------|-------------|
| `improvement_rate` | Percentage of responses that required improvement |
| `safety_improvement_rate` | Percentage of responses improved for safety reasons |
| `completeness_improvement_rate` | Percentage of responses improved for completeness |
| `empathy_improvement_rate` | Percentage of responses improved for empathy |

## Viewing Metrics

Metrics can be accessed in several ways:

1. **Admin Dashboard**: Navigate to `/admin/dashboard` in your browser
2. **JSON Files**: Examine the JSON test result files
3. **API Endpoint**: Use the `/api/metrics` endpoint to get raw metrics
4. **Logs**: Check the evaluation logs for detailed trace information

## Using Metrics for Improvement

Here's how to use these metrics to improve the chatbot:

1. **High Inference Time**: If certain topics consistently show high inference times, consider optimizing the retrieval process or caching common responses.

2. **Low ROUGE Scores**: Indicates potential issues with response quality. Consider improving the RAG corpus or fine-tuning the response generation process.

3. **Low Context Precision/Recall**: Suggests retrieval issues. Improve vector embeddings, expand the knowledge base, or adjust retrieval algorithms.

4. **Low Citation Rates**: If citations are frequently missing, adjust the threshold for when to add trusted sources or expand the list of trusted sources.

5. **Poor Topic Detection**: If topic detection accuracy is low for specific topics, add more examples or keywords for those topics.

6. **Negative User Feedback**: Analyze patterns in negative feedback to identify areas for improvement, particularly for specific topics or question types.