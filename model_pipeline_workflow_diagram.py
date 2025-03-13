import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Set up the figure with an ML/AI theme background
plt.figure(figsize=(16, 12))
ax = plt.gca()
ax.set_facecolor('#F5F7F9')  # Light blue-gray background

# Define modern ML/AI-themed colors
primary_color = '#6236FF'     # Deep purple for main model components
bert_color = '#05C3DD'        # Teal blue for BERT components
gpt_color = '#7B61FF'         # Purple for GPT components
policy_color = '#4CAF50'      # Green for policy components
eval_color = '#FFC107'        # Yellow/gold for evaluation
safety_color = '#FF5252'      # Red for safety checks
data_color = '#26A69A'        # Teal for data processing
input_color = '#3F51B5'       # Indigo for input processing

# Function to create a styled box with optional icon
def create_box(x, y, width, height, label, color, alpha=0.9, fontsize=11, icon=None):
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=patches.BoxStyle("Round", pad=0.6, rounding_size=0.2),
        facecolor=color, alpha=alpha, edgecolor='#333333', linewidth=1
    )
    ax.add_patch(rect)
    
    # Just use the label without icons
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
            fontsize=fontsize, color='white', fontweight='bold', 
            wrap=True, multialignment="center")
                
    return (x + width/2, y + height/2)

# Function to create a connector arrow
def create_arrow(start, end, color='#333333', style='->', width=1.5, connectionstyle="arc3,rad=0.1"):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=width,
                                connectionstyle=connectionstyle))

# Function to create a process arrow with label
def create_process_arrow(start, end, label, color='#333333', offset=(0, 0), fontsize=9, connectionstyle="arc3,rad=0.1"):
    midx = (start[0] + end[0]) / 2 + offset[0]
    midy = (start[1] + end[1]) / 2 + offset[1]
    create_arrow(start, end, color=color, connectionstyle=connectionstyle)
    ax.text(midx, midy, label, ha='center', va='center', fontsize=fontsize,
            bbox=dict(facecolor='white', alpha=0.9, edgecolor=color, boxstyle='round,pad=0.2'),
            color='#333333')

# ----- Input Processing -----
input_pos = create_box(1, 9, 2.5, 1, "User Question\nInput", input_color, icon="❓")
preprocess_pos = create_box(4.5, 9, 2.5, 1, "Text\nPreprocessing", input_color, icon="🔍")

# ----- Question Analysis -----
pii_pos = create_box(4.5, 7, 2.5, 1, "PII Detection", safety_color, icon="🔒")
category_pos = create_box(8, 9, 2.5, 1, "Question\nCategorization", primary_color, icon="🏷️")

# ----- Knowledge Path -----
bert_model_pos = create_box(1, 5, 2.5, 1, "BERT Embeddings", bert_color, icon="🧠")
vector_db_pos = create_box(1, 3, 2.5, 1, "Vector Database\nSearch", bert_color, icon="🔎")
knowledge_pos = create_box(1, 1, 2.5, 1, "Knowledge\nRetrieval & Ranking", bert_color, icon="📚")

# ----- Conversational Path -----
gpt_pos = create_box(4.5, 5, 2.5, 1, "GPT-4 Context\nPreparation", gpt_color, icon="💬")
prompt_pos = create_box(4.5, 3, 2.5, 1, "Prompt\nEngineering", gpt_color, icon="✏️")
convo_pos = create_box(4.5, 1, 2.5, 1, "Conversational\nResponse Generation", gpt_color, icon="🗣️")

# ----- Policy Path -----
location_pos = create_box(8, 7, 2.5, 1, "Location Context\nExtraction", policy_color, icon="📍")
policy_api_pos = create_box(8, 5, 2.5, 1, "Policy API\nInterface", policy_color, icon="📋")
policy_data_pos = create_box(8, 3, 2.5, 1, "Policy Data\nProcessing", policy_color, icon="⚖️")
policy_pos = create_box(8, 1, 2.5, 1, "State-Specific\nPolicy Response", policy_color, icon="🏛️")

# ----- Evaluation & Enhancement -----
response_pos = create_box(11.5, 9, 2.5, 1, "Response\nSelection", primary_color, icon="✅")
eval_pos = create_box(11.5, 7, 2.5, 1, "Response\nEvaluation", eval_color, icon="⭐")
safety_pos = create_box(11.5, 5, 2.5, 1, "Safety & Quality\nChecks", safety_color, icon="🛡️")
citation_pos = create_box(11.5, 3, 2.5, 1, "Citation\nManagement", data_color, icon="📝")
final_pos = create_box(11.5, 1, 2.5, 1, "Final Response\nFormatting", primary_color, icon="📤")

# ----- Connect the Components -----

# Input flow
create_process_arrow(input_pos, preprocess_pos, "Clean & Normalize")
create_process_arrow(preprocess_pos, pii_pos, "Detect & Sanitize PII", connectionstyle="arc3,rad=0.3")
create_process_arrow(preprocess_pos, category_pos, "Analyze Intent")

# Categorization to paths
create_process_arrow(category_pos, bert_model_pos, "Knowledge\nQuestion", connectionstyle="arc3,rad=-0.3")
create_process_arrow(category_pos, gpt_pos, "Conversational\nQuestion", connectionstyle="arc3,rad=-0.1")
create_process_arrow(category_pos, policy_api_pos, "Policy\nQuestion", connectionstyle="arc3,rad=0.1")

# Knowledge path flow
create_process_arrow(bert_model_pos, vector_db_pos, "Generate\nEmbeddings")
create_process_arrow(vector_db_pos, knowledge_pos, "Retrieve\nSimilar Q&A")

# Conversational path flow
create_process_arrow(gpt_pos, prompt_pos, "Build\nContext")
create_process_arrow(prompt_pos, convo_pos, "Optimize\nPrompt")

# Policy path flow
create_process_arrow(pii_pos, location_pos, "Extract Location")
create_process_arrow(location_pos, policy_api_pos, "State Context")
create_process_arrow(policy_api_pos, policy_data_pos, "API Request")
create_process_arrow(policy_data_pos, policy_pos, "Format Policy\nData")

# Response paths to selection
create_process_arrow(knowledge_pos, response_pos, "Knowledge\nResponse", connectionstyle="arc3,rad=0.4")
create_process_arrow(convo_pos, response_pos, "Conversational\nResponse", connectionstyle="arc3,rad=0.3")
create_process_arrow(policy_pos, response_pos, "Policy\nResponse", connectionstyle="arc3,rad=0.2")

# Evaluation and enhancement flow
create_process_arrow(response_pos, eval_pos, "Evaluate\nQuality")
create_process_arrow(eval_pos, safety_pos, "Check\nSafety")
create_process_arrow(safety_pos, citation_pos, "Add Citations\n& Sources")
create_process_arrow(citation_pos, final_pos, "Format for\nDisplay")

# ----- Add Legend for Component Types -----
legend_items = [
    (primary_color, "Core Processing"),
    (bert_color, "BERT RAG Knowledge Path"),
    (gpt_color, "GPT Conversational Path"),
    (policy_color, "Policy Information Path"),
    (eval_color, "Evaluation & Quality"),
    (safety_color, "Safety & PII Protection"),
    (input_color, "Input Processing"),
]

for i, (color, label) in enumerate(legend_items):
    rect = patches.Rectangle((14, 9-i*0.5), 0.5, 0.3, facecolor=color, edgecolor='#333333')
    ax.add_patch(rect)
    ax.text(14.7, 9-i*0.5+0.15, label, va='center', fontsize=9, color='#333333')

# ----- Add Pipeline Checks Annotations -----
checks_boxes = [
    (13.5, 6.3, 3, 1.2, "Response Quality Checks", "#FFC107", 0.1),
    (13.5, 4.3, 3, 1.2, "Safety & PII Checks", "#FF5252", 0.1),
    (13.5, 2.3, 3, 1.2, "Model Accuracy Checks", "#6236FF", 0.1),
]

for x, y, w, h, label, color, alpha in checks_boxes:
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=patches.BoxStyle("Round", pad=0.6, rounding_size=0.2),
        facecolor=color, alpha=alpha, edgecolor=color, linewidth=1
    )
    ax.add_patch(box)
    ax.text(x + 0.2, y + h - 0.3, label, fontsize=10, fontweight='bold', color='#333333')

checks_info = [
    (13.7, 6.2, "• Relevance Score (>7.0/10)", "#333333"),
    (13.7, 6.0, "• Empathy Factor Check", "#333333"),
    (13.7, 5.8, "• Source Citation Validation", "#333333"),
    (13.7, 5.6, "• Content Completeness", "#333333"),
    
    (13.7, 4.2, "• PII Detection & Redaction", "#333333"),
    (13.7, 4.0, "• Content Toxicity Scan (<0.3)", "#333333"),
    (13.7, 3.8, "• Harmful Advice Detection", "#333333"),
    (13.7, 3.6, "• Medical Disclaimer Check", "#333333"),
    
    (13.7, 2.2, "• Confidence Threshold (>0.7)", "#333333"),
    (13.7, 2.0, "• False Information Detection", "#333333"),
    (13.7, 1.8, "• Source Authority Verification", "#333333"),
    (13.7, 1.6, "• Hallucination Prevention", "#333333"),
]

for x, y, text, color in checks_info:
    ax.text(x, y, text, fontsize=9, color=color)

# Add title
plt.title('Reproductive Health Chatbot Model Pipeline Workflow\nComponent Integration & Quality Checks', fontsize=16, y=1.02, color='#333333')

# Remove axis
plt.axis('off')
plt.tight_layout()

# Save the figure
plt.savefig('model_pipeline_workflow_diagram.png', dpi=300, bbox_inches='tight', facecolor='#F5F7F9')
plt.close()

print("Model pipeline workflow diagram created successfully! Saved as 'model_pipeline_workflow_diagram.png'")