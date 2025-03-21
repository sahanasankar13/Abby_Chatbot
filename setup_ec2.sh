#!/bin/bash

# Update system
echo "Updating system packages..."
sudo dnf update -y

# Install Python and required packages
echo "Installing Python and dependencies..."
sudo dnf install -y python3 python3-pip python3-devel git

# Install PostgreSQL and other dependencies
echo "Installing PostgreSQL and other dependencies..."
sudo dnf install -y postgresql postgresql-devel gcc

# Create application directory
echo "Setting up application directory..."
mkdir -p ~/abby-chatbot

# Unzip application
echo "Extracting application files..."
unzip -q ~/abby-chatbot.zip -d ~/abby-chatbot

# Move .env file to the application directory
echo "Setting up environment variables..."
mv ~/.env ~/abby-chatbot/.env

# Change to application directory
cd ~/abby-chatbot

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install requirements
echo "Installing Python dependencies (this may take a while)..."
pip install wheel
pip install -r requirements.txt

# Run NLTK data downloader
echo "Downloading NLTK data..."
python download_nltk_data.py

# Create a systemd service file
echo "Setting up systemd service..."
cat << 'EOT' | sudo tee /etc/systemd/system/abby-chatbot.service
[Unit]
Description=Abby Chatbot Flask Application
After=network.target

[Service]
User=ec2-user
WorkingDirectory=/home/ec2-user/abby-chatbot
ExecStart=/home/ec2-user/abby-chatbot/venv/bin/gunicorn --worker-class gthread --workers 3 --bind 0.0.0.0:5000 --timeout 120 app:app
Restart=always
Environment="PYTHONUNBUFFERED=1"

[Install]
WantedBy=multi-user.target
EOT

# Enable and start the service
echo "Starting the application service..."
sudo systemctl daemon-reload
sudo systemctl enable abby-chatbot
sudo systemctl start abby-chatbot

# Print service status
echo "Service status:"
sudo systemctl status abby-chatbot

echo "Setup complete! The application should be running at http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4):5000" 