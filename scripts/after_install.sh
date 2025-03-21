#!/bin/bash

# After Install script for AWS CodeDeploy
# This script runs after the application is installed

# Exit on error
set -e

echo "Starting After Install process..."

# Set directory permissions
echo "Setting permissions..."
chown -R ec2-user:ec2-user /var/www/html
chmod -R 755 /var/www/html

# Activate virtual environment and install dependencies
echo "Installing application dependencies..."
cd /var/www/html
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-aws.txt

# Make scripts executable
echo "Making scripts executable..."
chmod +x /var/www/html/scripts/*.sh

# Ensure logs directory exists
mkdir -p /var/www/html/logs

# Set up systemd service for the application
echo "Setting up systemd service..."
cat > /etc/systemd/system/reproductive-health-chatbot.service << 'EOL'
[Unit]
Description=Reproductive Health Chatbot
After=network.target

[Service]
User=ec2-user
Group=ec2-user
WorkingDirectory=/var/www/html
Environment="PATH=/var/www/html/venv/bin"
ExecStart=/var/www/html/venv/bin/gunicorn --config=/var/www/html/gunicorn.conf.py app:app
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=reproductive-health-chatbot

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd
systemctl daemon-reload

echo "After Install completed successfully"