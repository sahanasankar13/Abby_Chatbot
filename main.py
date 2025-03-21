import os
import nltk
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Download required NLTK data packages
logger.info("Downloading required NLTK data packages...")
nltk_packages = ['punkt', 'stopwords']
for package in nltk_packages:
    try:
        nltk.download(package, quiet=True)
        logger.info(f"Successfully downloaded NLTK package: {package}")
    except Exception as e:
        logger.error(f"Error downloading NLTK package {package}: {str(e)}")

# Import app after NLTK data is downloaded
from app import app

if __name__ == "__main__":
    # Use port from environment variable or default to 5000 for Replit
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)