#!/bin/bash

# Variables
EC2_IP="3.94.80.205"
KEY_FILE="abby-chatbot-key.pem"
APP_NAME="abby-chatbot"
ZIP_FILE="${APP_NAME}.zip"

# Create a zip file of the application
echo "Creating zip file of the application..."
zip -r ${ZIP_FILE} . -x "*.git*" "*.DS_Store" "*.pem" "venv/*" ".venv/*" "__pycache__/*" "*.zip"

# Transfer the zip file to the EC2 instance
echo "Transferring zip file to EC2 instance..."
scp -i ${KEY_FILE} -o StrictHostKeyChecking=no ${ZIP_FILE} ec2-user@${EC2_IP}:~/${ZIP_FILE}

# Transfer .env file separately for security
echo "Transferring .env file securely..."
scp -i ${KEY_FILE} -o StrictHostKeyChecking=no .env ec2-user@${EC2_IP}:~/abby-chatbot.env

# SSH into the EC2 instance and set up the application
echo "Setting up the application on the EC2 instance..."
ssh -i ${KEY_FILE} -o StrictHostKeyChecking=no ec2-user@${EC2_IP} << 'EOF'
  # Update system
  sudo dnf update -y
  
  # Install Python and required packages
  sudo dnf install -y python3 python3-pip python3-devel git
  
  # Install PostgreSQL
  sudo dnf install -y postgresql postgresql-devel
  
  # Create application directory
  mkdir -p ~/abby-chatbot
  
  # Unzip application
  unzip ~/abby-chatbot.zip -d ~/abby-chatbot
  
  # Move the .env file to the application directory
  mv ~/abby-chatbot.env ~/abby-chatbot/.env
  
  # Change to application directory
  cd ~/abby-chatbot
  
  # Create virtual environment
  python3 -m venv venv
  source venv/bin/activate
  
  # Install requirements
  pip install -r requirements.txt
  
  # Run NLTK data downloader if needed
  python download_nltk_data.py
  
  # Create a systemd service file
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
  sudo systemctl daemon-reload
  sudo systemctl enable abby-chatbot
  sudo systemctl start abby-chatbot
  
  # Print service status
  echo "Service status:"
  sudo systemctl status abby-chatbot
EOF

echo "Deployment complete! The application should be running at http://${EC2_IP}:5000" 