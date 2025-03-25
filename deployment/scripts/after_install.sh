#!/bin/bash

# Exit on error
set -e

# Logging configuration
exec > >(tee /var/log/codedeploy-after-install.log) 2>&1
echo "Starting AfterInstall script at $(date)"

# Set environment variables
APP_DIR="/opt/abby-chatbot"
VENV_DIR="$APP_DIR/venv"
DATA_DIR="$APP_DIR/data"
MODELS_DIR="$APP_DIR/serialized_models"
NLTK_DIR="$APP_DIR/nltk_data"
export NLTK_DATA=$NLTK_DIR

# Activate virtual environment
source $VENV_DIR/bin/activate

# Install requirements
pip install -r $APP_DIR/requirements.txt

# Install additional packages needed for deployment
pip install gunicorn

# Download required NLTK data
python3 -c "import nltk; nltk.download('punkt', download_dir='$NLTK_DIR'); nltk.download('averaged_perceptron_tagger', download_dir='$NLTK_DIR'); nltk.download('wordnet', download_dir='$NLTK_DIR')"

# Run data setup script
chmod +x $APP_DIR/scripts/setup_data.sh
$APP_DIR/scripts/setup_data.sh

# Set correct permissions
chown -R ec2-user:ec2-user $APP_DIR
chmod -R 755 $APP_DIR

echo "AfterInstall script completed successfully"
exit 0