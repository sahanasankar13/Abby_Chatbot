#!/bin/bash

# Abort on error with detailed output
set -e

# Default values
KEY_NAME="abby-chatbot-key.pem"
INSTANCE_ID=""
SERVER_USER="ec2-user"
SERVER_IP=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        --key)
            KEY_NAME="$2"
            shift 2
            ;;
        --instance)
            INSTANCE_ID="$2"
            shift 2
            ;;
        --ip)
            SERVER_IP="$2"
            shift 2
            ;;
        --user)
            SERVER_USER="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --key KEY_FILE       SSH key file (default: abby-chatbot-key.pem)"
            echo "  --instance ID        EC2 instance ID"
            echo "  --ip IP              Server IP address"
            echo "  --user USER          Server username (default: ec2-user)"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required parameters
if [ -z "$SERVER_IP" ] && [ -z "$INSTANCE_ID" ]; then
    echo "Error: Either --ip or --instance must be provided"
    exit 1
fi

# If instance ID is provided but IP is not, get the IP
if [ -n "$INSTANCE_ID" ] && [ -z "$SERVER_IP" ]; then
    echo "Getting public IP for instance $INSTANCE_ID..."
    SERVER_IP=$(aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
        --query "Reservations[0].Instances[0].PublicIpAddress" --output text)
    
    if [ -z "$SERVER_IP" ] || [ "$SERVER_IP" == "None" ]; then
        echo "Error: Could not get public IP for instance $INSTANCE_ID"
        exit 1
    fi
    
    echo "Instance public IP: $SERVER_IP"
fi

# Check if key file exists and has proper permissions
if [ ! -f "$KEY_NAME" ]; then
    echo "Error: SSH key file $KEY_NAME not found"
    exit 1
fi

# Ensure key has proper permissions
chmod 400 "$KEY_NAME"

echo "Preparing to deploy Abby Chatbot to $SERVER_IP..."

# Create deployment package
echo "Creating deployment package..."
APP_ZIP="abby-chatbot.zip"
zip -r "$APP_ZIP" app.py models.py requirements.txt templates static chatbot utils .env data config \
    download_nltk_data.py setup_ec2.sh models nltk_data -x "**/__pycache__/*" "**/.DS_Store" "**/.git/*"

# Wait for SSH to be available (with timeout)
echo "Waiting for SSH connection..."
MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY_NAME" $SERVER_USER@$SERVER_IP echo "SSH connection successful"; then
        echo "SSH connection established!"
        break
    fi
    
    echo "SSH connection failed. Retrying in 10 seconds... (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)"
    RETRY_COUNT=$((RETRY_COUNT+1))
    sleep 10
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "Error: Could not establish SSH connection after $MAX_RETRIES attempts"
    exit 1
fi

# Upload files
echo "Uploading application files..."
scp -i "$KEY_NAME" "$APP_ZIP" $SERVER_USER@$SERVER_IP:~/ || { echo "Error uploading app package"; exit 1; }
scp -i "$KEY_NAME" .env $SERVER_USER@$SERVER_IP:~/ || { echo "Error uploading .env file"; exit 1; }
scp -i "$KEY_NAME" setup_ec2.sh $SERVER_USER@$SERVER_IP:~/ || { echo "Error uploading setup script"; exit 1; }

# Execute setup script
echo "Setting up the server..."
ssh -i "$KEY_NAME" $SERVER_USER@$SERVER_IP "chmod +x setup_ec2.sh && ./setup_ec2.sh"

# Create a systemd service file to run the app
echo "Creating systemd service..."
cat > abby-chatbot.service << EOF
[Unit]
Description=Abby Chatbot Flask Application
After=network.target

[Service]
User=$SERVER_USER
WorkingDirectory=/home/$SERVER_USER/abby-chatbot
ExecStart=/home/$SERVER_USER/abby-chatbot/venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Upload and enable service
scp -i "$KEY_NAME" abby-chatbot.service $SERVER_USER@$SERVER_IP:~/
ssh -i "$KEY_NAME" $SERVER_USER@$SERVER_IP "sudo mv ~/abby-chatbot.service /etc/systemd/system/ && \
    sudo systemctl daemon-reload && \
    sudo systemctl enable abby-chatbot && \
    sudo systemctl start abby-chatbot"

# Check if service is running
echo "Checking if service is running..."
ssh -i "$KEY_NAME" $SERVER_USER@$SERVER_IP "sudo systemctl status abby-chatbot"

# Display application URL
echo "Deployment completed!"
echo "Your application should be accessible at: http://$SERVER_IP:5000"

# Security group configuration reminder
echo "Remember to ensure your EC2 security group allows traffic on port 5000" 