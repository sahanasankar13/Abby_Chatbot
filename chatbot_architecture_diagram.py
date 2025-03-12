
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Set up the figure with a light background
plt.figure(figsize=(16, 12))
ax = plt.gca()
ax.set_facecolor('#f8f9fa')

# Define colors
primary_color = '#3498db'  # Main components
secondary_color = '#2ecc71'  # Processing steps
warning_color = '#e74c3c'  # Evaluation/safety
info_color = '#9b59b6'  # Data sources
neutral_color = '#95a5a6'  # Supporting components

# Function to create a styled box
def create_box(x, y, width, height, label, color, alpha=0.9, fontsize=10):
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=patches.BoxStyle("Round", pad=0.6, rounding_size=0.2),
        facecolor=color, alpha=alpha, edgecolor='black', linewidth=1
    )
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
            fontsize=fontsize, color='black', fontweight='bold', 
            wrap=True, multialignment="center")
    return (x + width/2, y + height/2)

# Function to create a connector arrow
def create_arrow(start, end, color='black', style='->', width=1.5, connectionstyle="arc3,rad=0.1"):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=width,
                                connectionstyle=connectionstyle))

# Function to create a process arrow with label
def create_process_arrow(start, end, label, color='black', offset=(0, 0), fontsize=8):
    midx = (start[0] + end[0]) / 2 + offset[0]
    midy = (start[1] + end[1]) / 2 + offset[1]
    create_arrow(start, end, color=color)
    ax.text(midx, midy, label, ha='center', va='center', fontsize=fontsize,
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

# ----- Main Components -----

# User Interface
ui_pos = create_box(1, 9, 2, 1, "User Interface", primary_color)

# Conversation Manager
cm_pos = create_box(5, 9, 3, 1, "Conversation Manager", primary_color)

# Baseline Model
bm_pos = create_box(5, 7, 3, 1, "Baseline Model", primary_color)

# ----- Category Classification -----
cat_pos = create_box(5, 5, 3, 1, "Question Categorization", secondary_color)

# ----- Response Generation Components -----
bert_pos = create_box(1, 3, 2, 1, "BERT RAG Model\n(Knowledge)", info_color)
gpt_pos = create_box(4, 3, 2, 1, "GPT Model\n(Conversational)", info_color)
policy_pos = create_box(7, 3, 2, 1, "Policy API\n(Policy Data)", info_color)

# ----- Evaluation and Enhancement -----
eval_pos = create_box(10, 5, 3, 1, "Response Evaluator", warning_color)
friendly_pos = create_box(10, 7, 3, 1, "Friendly Bot", neutral_color)
citation_pos = create_box(10, 3, 3, 1, "Citation Manager", neutral_color)

# ----- Safety Checks -----
safety_pos = create_box(10, 1, 3, 1, "Safety & Quality\nChecks", warning_color)

# ----- Final Response -----
final_pos = create_box(5, 1, 3, 1, "Final Response", secondary_color)

# ----- Connect the Components -----

# User to Conversation Manager
create_process_arrow(ui_pos, cm_pos, "User Question")

# Conversation Manager to Baseline
create_process_arrow(cm_pos, bm_pos, "Process Question")

# Baseline to Categorization
create_process_arrow(bm_pos, cat_pos, "Categorize")

# Categorization to Response Components
create_process_arrow(cat_pos, bert_pos, "Knowledge", 
                    offset=(-0.5, 0), connectionstyle="arc3,rad=-0.2")
create_process_arrow(cat_pos, gpt_pos, "Conversational", 
                    offset=(0, -0.3))
create_process_arrow(cat_pos, policy_pos, "Policy", 
                    offset=(0.5, 0), connectionstyle="arc3,rad=0.2")

# Component outputs to Evaluation
create_process_arrow(bert_pos, eval_pos, "RAG Response", 
                    connectionstyle="arc3,rad=0.3")
create_process_arrow(gpt_pos, eval_pos, "GPT Response", 
                    connectionstyle="arc3,rad=0.2")
create_process_arrow(policy_pos, eval_pos, "Policy Response", 
                    connectionstyle="arc3,rad=0.1")

# Evaluation to Safety
create_process_arrow(eval_pos, safety_pos, "Evaluate Response")

# Safety to Final
create_process_arrow(safety_pos, final_pos, "Safety Check")

# Enhancement flow
create_process_arrow(final_pos, friendly_pos, "Enhance", 
                    connectionstyle="arc3,rad=0.3")
create_process_arrow(friendly_pos, citation_pos, "Add Empathy", 
                    connectionstyle="arc3,rad=0.3")
create_process_arrow(citation_pos, ui_pos, "Add Citations", 
                    connectionstyle="arc3,rad=0.3")

# ----- Add Legend for Process Types -----
legend_items = [
    (primary_color, "Core Components"),
    (secondary_color, "Processing Steps"),
    (info_color, "Data Sources"),
    (warning_color, "Evaluation & Safety"),
    (neutral_color, "Enhancement")
]

for i, (color, label) in enumerate(legend_items):
    rect = patches.Rectangle((13, 9-i*0.5), 0.5, 0.3, facecolor=color)
    ax.add_patch(rect)
    ax.text(13.7, 9-i*0.5+0.15, label, va='center', fontsize=9)

# ----- Add Labels for Major Processes -----

# Process areas
ax.text(13, 7, "1. Question Analysis", fontsize=10, fontweight='bold')
ax.text(13, 6.7, "- Extract location context", fontsize=8)
ax.text(13, 6.4, "- Detect emotional content", fontsize=8)
ax.text(13, 6.1, "- Categorize question type", fontsize=8)

ax.text(13, 5.5, "2. Response Generation", fontsize=10, fontweight='bold')
ax.text(13, 5.2, "- BERT RAG for knowledge", fontsize=8)
ax.text(13, 4.9, "- GPT for conversation", fontsize=8)
ax.text(13, 4.6, "- Policy API for regulations", fontsize=8)

ax.text(13, 4.0, "3. Safety & Quality", fontsize=10, fontweight='bold')
ax.text(13, 3.7, "- Toxicity detection", fontsize=8)
ax.text(13, 3.4, "- Source validation", fontsize=8)
ax.text(13, 3.1, "- Content improvement", fontsize=8)

ax.text(13, 2.5, "4. Enhancement", fontsize=10, fontweight='bold')
ax.text(13, 2.2, "- Add empathetic elements", fontsize=8)
ax.text(13, 1.9, "- Improve structure", fontsize=8)
ax.text(13, 1.6, "- Add proper citations", fontsize=8)

# Add title
plt.title('Reproductive Health Chatbot Architecture\nProcessing Flow and Checks', fontsize=14, y=1.02)

# Remove axis
plt.axis('off')
plt.tight_layout()

# Save the figure
plt.savefig('chatbot_architecture_diagram.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("Architecture diagram created successfully! Saved as 'chatbot_architecture_diagram.png'")
