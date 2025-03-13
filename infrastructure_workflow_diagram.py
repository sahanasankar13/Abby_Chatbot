import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.path import Path

# Set up the figure with a dark blue background (AWS style)
plt.figure(figsize=(16, 12))
ax = plt.gca()
ax.set_facecolor('#232F3E')  # AWS dark blue background color

# Define AWS-inspired colors
aws_orange = '#FF9900'  # AWS primary color
aws_blue = '#00A1C9'    # AWS secondary blue
aws_teal = '#1DC2C2'    # Teal
aws_green = '#7FC942'   # Green for success/completion
aws_red = '#D13212'     # Red for alerts/errors
aws_purple = '#BD8BCC'  # Purple for ML services
aws_gray = '#687078'    # Gray for infrastructure

# Function to create a styled box
def create_box(x, y, width, height, label, color, alpha=0.9, fontsize=11, icon=None):
    rect = patches.FancyBboxPatch(
        (x, y), width, height,
        boxstyle=patches.BoxStyle("Round", pad=0.6, rounding_size=0.2),
        facecolor=color, alpha=alpha, edgecolor='white', linewidth=1
    )
    ax.add_patch(rect)
    
    # Just use the label without icons
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
            fontsize=fontsize, color='white', fontweight='bold', 
            wrap=True, multialignment="center")
                
    return (x + width/2, y + height/2)

# Function to create a connector arrow
def create_arrow(start, end, color='white', style='->', width=1.5, connectionstyle="arc3,rad=0.1"):
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=width,
                                connectionstyle=connectionstyle))

# Function to create a process arrow with label
def create_process_arrow(start, end, label, color='white', offset=(0, 0), fontsize=9, connectionstyle="arc3,rad=0.1"):
    midx = (start[0] + end[0]) / 2 + offset[0]
    midy = (start[1] + end[1]) / 2 + offset[1]
    create_arrow(start, end, color=color, connectionstyle=connectionstyle)
    ax.text(midx, midy, label, ha='center', va='center', fontsize=fontsize,
            bbox=dict(facecolor='#232F3E', alpha=0.8, edgecolor='white', boxstyle='round,pad=0.2'),
            color='white')

# ----- Client/User Layer -----
user_pos = create_box(1, 9.5, 2.5, 1, "End Users", aws_blue, icon="👥")
admin_pos = create_box(12.5, 9.5, 2.5, 1, "Admin Users", aws_blue, icon="👤")

# ----- Load Balancers -----
lb_pos = create_box(7, 9.5, 2, 0.8, "Elastic Load Balancer", aws_orange, icon="⚖️")

# ----- Web/App Layer -----
ec2_web1_pos = create_box(2, 7.5, 2, 0.8, "EC2 Instance\nWeb Tier (AZ1)", aws_orange, icon="🖥️")
ec2_web2_pos = create_box(5, 7.5, 2, 0.8, "EC2 Instance\nWeb Tier (AZ2)", aws_orange, icon="🖥️")
ec2_admin_pos = create_box(12, 7.5, 2, 0.8, "EC2 Instance\nAdmin Tier", aws_orange, icon="🖥️")

# ----- Application Layer -----
app_layer_pos = create_box(3.5, 5.5, 3, 1, "Chatbot Application\nContainer Cluster", aws_teal, icon="🐳")
monitoring_pos = create_box(12, 5.5, 2, 1, "Monitoring &\nLogging", aws_red, icon="📊")

# ----- Database Layer -----
primary_db_pos = create_box(2, 3, 2, 1, "Primary DB\nInstance", aws_blue, icon="💾")
replica_db_pos = create_box(5, 3, 2, 1, "Read Replica\nDB Instance", aws_blue, icon="💾")
metrics_db_pos = create_box(8, 3, 2, 1, "Metrics\nDatabase", aws_blue, icon="📈")

# ----- Storage -----
s3_logs_pos = create_box(12, 3, 2, 1, "S3 Bucket\n(Logs & Backups)", aws_orange, icon="🗄️")

# ----- ML Infrastructure -----
sageMaker_pos = create_box(3.5, 1.5, 3, 1, "SageMaker\nModel Hosting", aws_purple, icon="🧠")

# ----- Security Components -----
waf_pos = create_box(8, 9.5, 2, 0.8, "AWS WAF", aws_red, icon="🛡️")
sec_group_pos = create_box(8, 7.5, 2, 0.8, "Security Groups", aws_red, icon="🔒")
iam_pos = create_box(8, 5.5, 2, 1, "IAM &\nSecrets Manager", aws_red, icon="🔑")

# ----- Connect the Components -----

# Users to Load Balancer & WAF
create_process_arrow(user_pos, waf_pos, "HTTPS Requests")
create_process_arrow(admin_pos, waf_pos, "Admin HTTPS Requests", connectionstyle="arc3,rad=-0.1")

# WAF to Load Balancer
create_process_arrow(waf_pos, lb_pos, "Filtered Traffic")

# Load Balancer to Web Tier
create_process_arrow(lb_pos, ec2_web1_pos, "Route Traffic\n(Primary AZ)", offset=(-0.5, 0), connectionstyle="arc3,rad=-0.2")
create_process_arrow(lb_pos, ec2_web2_pos, "Route Traffic\n(Secondary AZ)", offset=(0.5, 0), connectionstyle="arc3,rad=0.2")
create_process_arrow(lb_pos, ec2_admin_pos, "Route Admin\nTraffic", connectionstyle="arc3,rad=0.3")

# Security Groups
create_process_arrow(sec_group_pos, ec2_web1_pos, "Enforce\nSecurity", color=aws_red, connectionstyle="arc3,rad=-0.2")
create_process_arrow(sec_group_pos, ec2_web2_pos, "Enforce\nSecurity", color=aws_red, connectionstyle="arc3,rad=-0.1")
create_process_arrow(sec_group_pos, ec2_admin_pos, "Enforce\nSecurity", color=aws_red, connectionstyle="arc3,rad=0.2")

# Web Tier to App Layer
create_process_arrow(ec2_web1_pos, app_layer_pos, "Handle\nRequests", connectionstyle="arc3,rad=-0.1")
create_process_arrow(ec2_web2_pos, app_layer_pos, "Handle\nRequests", connectionstyle="arc3,rad=0.1")

# Admin Tier to Monitoring
create_process_arrow(ec2_admin_pos, monitoring_pos, "Monitor\nSysteme", connectionstyle="arc3,rad=0")

# App Layer to Databases
create_process_arrow(app_layer_pos, primary_db_pos, "Write\nOperations", connectionstyle="arc3,rad=-0.1")
create_process_arrow(app_layer_pos, replica_db_pos, "Read\nOperations", connectionstyle="arc3,rad=0.1")
create_process_arrow(app_layer_pos, metrics_db_pos, "Store\nMetrics", connectionstyle="arc3,rad=0.2")

# Database Replication
create_process_arrow(primary_db_pos, replica_db_pos, "Replicate\nData", color=aws_blue)

# App Layer to ML Infrastructure
create_process_arrow(app_layer_pos, sageMaker_pos, "Model\nInference", connectionstyle="arc3,rad=-0.1")

# Monitoring to Storage
create_process_arrow(monitoring_pos, s3_logs_pos, "Archive\nLogs")

# IAM connections
create_process_arrow(iam_pos, app_layer_pos, "Access\nControl", color=aws_red, connectionstyle="arc3,rad=-0.2")
create_process_arrow(iam_pos, sageMaker_pos, "Access\nControl", color=aws_red, connectionstyle="arc3,rad=0.3")

# Monitoring
create_process_arrow(app_layer_pos, monitoring_pos, "Log\nEvents", connectionstyle="arc3,rad=0.2")
create_process_arrow(sageMaker_pos, monitoring_pos, "Log\nModel Events", connectionstyle="arc3,rad=0.3")

# ----- Add Legend for Component Types -----
legend_items = [
    (aws_blue, "User Interfaces & Data Storage"),
    (aws_orange, "Compute & Networking"),
    (aws_teal, "Container Services"),
    (aws_purple, "ML Infrastructure"),
    (aws_red, "Security & Monitoring"),
]

for i, (color, label) in enumerate(legend_items):
    rect = patches.Rectangle((14, 1.5-i*0.4), 0.5, 0.3, facecolor=color, edgecolor='white')
    ax.add_patch(rect)
    ax.text(14.7, 1.5-i*0.4+0.15, label, va='center', fontsize=9, color='white')

# ----- Add Infrastructure Checks Annotations -----
checks_info = [
    (10.5, 8.5, "SECURITY CHECKS:", aws_red),
    (10.5, 8.2, "• WAF Rules for Request Filtering", 'white'),
    (10.5, 7.9, "• Security Group Rules Enforcement", 'white'),
    (10.5, 7.6, "• Secrets Rotation & Management", 'white'),
    
    (10.5, 7.0, "HIGH AVAILABILITY:", aws_green),
    (10.5, 6.7, "• Multi-AZ Deployment", 'white'),
    (10.5, 6.4, "• Load Balancer Health Checks", 'white'),
    (10.5, 6.1, "• DB Read Replica Failover", 'white'),
    
    (10.5, 5.5, "MONITORING & ALERTS:", aws_orange),
    (10.5, 5.2, "• Resource Utilization Metrics", 'white'),
    (10.5, 4.9, "• Error Rate & Latency Tracking", 'white'),
    (10.5, 4.6, "• Auto-scaling Triggers", 'white'),
    
    (10.5, 4.0, "COMPLIANCE CHECKS:", aws_blue),
    (10.5, 3.7, "• PII Data Handling Validation", 'white'),
    (10.5, 3.4, "• Data Encryption (In-transit & At-rest)", 'white'),
    (10.5, 3.1, "• Access Control & Audit Logs", 'white'),
    
    (10.5, 2.5, "DEPLOYMENT WORKFLOW:", aws_teal),
    (10.5, 2.2, "• CI/CD Pipeline Integration", 'white'),
    (10.5, 1.9, "• Blue/Green Deployment Strategy", 'white'),
    (10.5, 1.6, "• Rollback Capability", 'white'),
]

for x, y, text, color in checks_info:
    ax.text(x, y, text, fontsize=9, color=color, fontweight='bold' if 'CHECKS' in text or 'AVAILABILITY' in text or 'MONITORING' in text or 'COMPLIANCE' in text or 'DEPLOYMENT' in text else 'normal')

# Add title
plt.title('Reproductive Health Chatbot Infrastructure Workflow\nAWS Deployment Architecture & Security Checks', fontsize=16, y=1.02, color='white')

# Remove axis
plt.axis('off')
plt.tight_layout()

# Save the figure
plt.savefig('infrastructure_workflow_diagram.png', dpi=300, bbox_inches='tight', facecolor='#232F3E')
plt.close()

print("Infrastructure workflow diagram created successfully! Saved as 'infrastructure_workflow_diagram.png'")