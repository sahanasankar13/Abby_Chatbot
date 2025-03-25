#!/bin/bash
set -e

# Stop the service if it's running
if systemctl is-active --quiet abby-chatbot; then
    systemctl stop abby-chatbot
fi

# Disable the service
systemctl disable abby-chatbot

# Kill any running gunicorn processes
pkill gunicorn || true

echo "Abby Chatbot service stopped successfully" 