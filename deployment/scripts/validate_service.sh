#!/bin/bash

# Exit on error
set -e

# Logging configuration
exec > >(tee /var/log/codedeploy-validate-service.log) 2>&1
echo "Starting service validation script at $(date)"

# Check if service is running
if ! systemctl is-active --quiet abby-chatbot; then
    echo "Service is not running!"
    systemctl status abby-chatbot | cat
    exit 1
fi

# Wait for application to be ready
echo "Waiting for application to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:5006/health > /dev/null; then
        echo "Application is responding to health checks"
        exit 0
    fi
    echo "Attempt $i: Application not ready yet..."
    sleep 10
done

echo "Application failed to respond to health checks after 5 minutes"
exit 1 