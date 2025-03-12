import os
import requests
import json
import logging
import time
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def test_abortion_policy_api():
    """Test the abortion policy API with a sample state"""
    
    # Get API key from environment
    api_key = os.environ.get("POLICY_API_KEY") or os.environ.get("ABORTION_POLICY_API_KEY")
    
    if not api_key:
        logger.error("No API key found in environment variables")
        return
    
    # Mask API key for logging
    masked_key = api_key[:3] + "*" * (len(api_key) - 3) if api_key else "None"
    logger.info(f"Using API key: {masked_key}")
    
    # Test with California (CA)
    state_code = "CA"
    base_url = "https://api.abortionpolicyapi.com/v1"
    
    # Define endpoints to test
    endpoints = {
        "waiting_periods": "waiting_periods",
        "insurance_coverage": "insurance_coverage",
        "gestational_limits": "gestational_limits",
        "minors": "minors"
    }
    
    # First try with token in headers
    headers_token = {'token': api_key}
    logger.info("Attempting with 'token' in headers")
    
    success = False
    
    # Try first approach: token in headers
    for key, endpoint in endpoints.items():
        url = f"{base_url}/{endpoint}/states/{state_code}"
        logger.info(f"Making request to: {url}")
        
        try:
            time.sleep(0.5)  # Rate limiting
            response = requests.get(url, headers=headers_token)
            
            logger.info(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"{key} endpoint successful: {json.dumps(data, indent=2)}")
                success = True
            else:
                logger.warning(f"{key} endpoint failed: {response.text}")
                
        except Exception as e:
            logger.error(f"Exception during API call to {key}: {str(e)}")
    
    # If all failed, try with x-api-key
    if not success:
        logger.info("Trying with 'x-api-key' in headers instead")
        headers_x_api = {'x-api-key': api_key}
        
        for key, endpoint in endpoints.items():
            url = f"{base_url}/{endpoint}/states/{state_code}"
            logger.info(f"Making request to: {url} with x-api-key")
            
            try:
                time.sleep(0.5)  # Rate limiting
                response = requests.get(url, headers=headers_x_api)
                
                logger.info(f"Response status code: {response.status_code}")
                
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"{key} endpoint successful: {json.dumps(data, indent=2)}")
                    success = True
                else:
                    logger.warning(f"{key} endpoint failed: {response.text}")
                    
            except Exception as e:
                logger.error(f"Exception during API call to {key}: {str(e)}")
    
    return success

if __name__ == "__main__":
    success = test_abortion_policy_api()
    print(f"API test {'succeeded' if success else 'failed'}")