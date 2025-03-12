
import nltk
import os

# Create directory for NLTK data if it doesn't exist
os.makedirs('/home/runner/nltk_data', exist_ok=True)

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

print("NLTK resources downloaded successfully!")
