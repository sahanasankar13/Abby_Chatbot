#!/bin/bash

# Application Start script for AWS CodeDeploy
# This script starts the application

# Exit on error
set -e

echo "Starting application..."

# Start the application service
systemctl start reproductive-health-chatbot.service
systemctl enable reproductive-health-chatbot.service

# Check if the application is running
if systemctl is-active --quiet reproductive-health-chatbot.service; then
  echo "Application started successfully"
  
  # Display service status
  systemctl status reproductive-health-chatbot.service --no-pager
else
  echo "Failed to start application"
  
  # Display logs for debugging
  journalctl -u reproductive-health-chatbot.service --no-pager -n 50
  exit 1
fi

echo "Application Start completed successfully"