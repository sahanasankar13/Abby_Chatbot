#!/bin/bash

# Application Stop script for AWS CodeDeploy
# This script stops the application if it's running

echo "Stopping application if running..."

# Check if the service exists before trying to stop it
if systemctl list-unit-files | grep -q reproductive-health-chatbot.service; then
  # Check if the service is active before stopping
  if systemctl is-active --quiet reproductive-health-chatbot.service; then
    echo "Stopping reproductive-health-chatbot service..."
    systemctl stop reproductive-health-chatbot.service
    echo "Service stopped"
  else
    echo "Service is not running"
  fi
else
  echo "Service does not exist, nothing to stop"
fi

# Additional cleanup if needed
# This is useful for first-time deployments or if service file changes
if [ -f /etc/systemd/system/reproductive-health-chatbot.service ]; then
  echo "Disabling service..."
  systemctl disable reproductive-health-chatbot.service
fi

echo "Application Stop completed successfully"