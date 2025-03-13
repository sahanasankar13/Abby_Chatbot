
import os

def get_openai_api_key():
    """Get OpenAI API key from environment variables"""
    return os.environ.get("OPENAI_API_KEY")

def get_abortion_policy_api_key():
    """Get Abortion Policy API key from environment variables"""
    return os.environ.get("POLICY_API_KEY", "tA3Z3l6l35344")
