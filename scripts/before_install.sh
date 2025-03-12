#!/bin/bash

# Before Install script for AWS CodeDeploy
# This script runs before the application is installed

# Exit on error
set -e

# Update system packages
echo "Updating system packages..."
yum update -y

# Install Python and required dependencies
echo "Installing Python and dependencies..."
yum install -y python311 python311-pip python311-devel gcc

# Create application directory if it doesn't exist
if [ ! -d /var/www/html ]; then
  echo "Creating application directory..."
  mkdir -p /var/www/html
fi

# Create a Python virtual environment if it doesn't exist
if [ ! -d /var/www/html/venv ]; then
  echo "Creating Python virtual environment..."
  cd /var/www/html
  python3.11 -m venv venv
fi

# Clean up any previous deployment artifacts
echo "Cleaning up previous deployment..."
rm -rf /var/www/html/app.py
rm -rf /var/www/html/models.py
rm -rf /var/www/html/main.py
rm -rf /var/www/html/static
rm -rf /var/www/html/templates
rm -rf /var/www/html/utils
rm -rf /var/www/html/chatbot

echo "Before Install completed successfully"