#!/bin/bash

# Exit on error
set -e

# Logging configuration
exec > >(tee /var/log/codedeploy-start-application.log) 2>&1
echo "Starting application script at $(date)"

# Set environment variables
APP_DIR="/opt/abby-chatbot"
VENV_DIR="$APP_DIR/venv"
LOG_DIR="/var/log/abby-chatbot"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR
chown -R ec2-user:ec2-user $LOG_DIR

# Create systemd service file
cat > /etc/systemd/system/abby-chatbot.service << EOF
[Unit]
Description=Abby Chatbot Gunicorn Service
After=network.target

[Service]
User=ec2-user
Group=ec2-user
WorkingDirectory=/opt/abby-chatbot
Environment="PATH=/opt/abby-chatbot/venv/bin"
ExecStart=/opt/abby-chatbot/venv/bin/gunicorn -b 0.0.0.0:5006 app:app --workers 3 --log-file=/var/log/abby-chatbot/gunicorn.log
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and start service
systemctl daemon-reload
systemctl enable abby-chatbot
systemctl restart abby-chatbot

# Check service status
systemctl status abby-chatbot | cat

echo "Application start script completed successfully"
exit 0 