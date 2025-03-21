"""
S3 Model Loader

Provides utilities to download and cache ML models from S3
for serverless deployment environments.
"""

import os
import logging
import tempfile
import boto3
import hashlib
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class S3ModelLoader:
    """
    Utility class to download and cache ML models from S3
    for serverless deployment environments.
    """
    
    def __init__(self, bucket_name=None, stage='dev'):
        """
        Initialize the loader.
        
        Args:
            bucket_name (str): S3 bucket name to load models from
            stage (str): Deployment stage (dev, staging, prod)
        """
        self.bucket_name = bucket_name or os.environ.get('MODEL_BUCKET')
        self.stage = stage or os.environ.get('STAGE', 'dev')
        self.cache_dir = os.path.join(tempfile.gettempdir(), 'abby-chatbot-models')
        self.s3_client = None
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"S3ModelLoader initialized with bucket: {self.bucket_name}, stage: {self.stage}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _initialize_s3_client(self):
        """Initialize the S3 client if not already done"""
        if self.s3_client is None:
            self.s3_client = boto3.client('s3')
    
    def _get_cache_path(self, s3_key):
        """
        Generate a cache path for an S3 key.
        
        Args:
            s3_key (str): S3 key of the model file
            
        Returns:
            str: Local cache path
        """
        # Create a hash of the S3 key to use as the filename
        filename = hashlib.md5(s3_key.encode()).hexdigest()
        return os.path.join(self.cache_dir, filename)
    
    def download_model(self, model_path, force_download=False):
        """
        Download a model from S3 and cache it locally.
        
        Args:
            model_path (str): Path to the model file or directory
            force_download (bool): Force download even if cached
            
        Returns:
            str: Path to the cached model
        """
        if not self.bucket_name:
            logger.warning("No S3 bucket specified, cannot download model")
            return model_path
        
        self._initialize_s3_client()
        
        # Construct S3 key
        s3_key = f"models/{self.stage}/{model_path}"
        cache_path = self._get_cache_path(s3_key)
        
        # Check if model is already cached
        if os.path.exists(cache_path) and not force_download:
            logger.info(f"Model already cached at {cache_path}")
            return cache_path
        
        try:
            logger.info(f"Downloading model from s3://{self.bucket_name}/{s3_key} to {cache_path}")
            self.s3_client.download_file(self.bucket_name, s3_key, cache_path)
            logger.info(f"Model downloaded successfully to {cache_path}")
            return cache_path
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            # Return original path as fallback
            return model_path
    
    def download_folder(self, folder_path, force_download=False):
        """
        Download a folder of model files from S3 and cache them locally.
        
        Args:
            folder_path (str): Path to the model folder
            force_download (bool): Force download even if cached
            
        Returns:
            str: Path to the cached model folder
        """
        if not self.bucket_name:
            logger.warning("No S3 bucket specified, cannot download folder")
            return folder_path
        
        self._initialize_s3_client()
        
        # Construct S3 prefix
        s3_prefix = f"models/{self.stage}/{folder_path}/"
        cache_folder = os.path.join(self.cache_dir, hashlib.md5(s3_prefix.encode()).hexdigest())
        
        # Check if folder is already cached
        if os.path.exists(cache_folder) and not force_download:
            logger.info(f"Folder already cached at {cache_folder}")
            return cache_folder
        
        # Create cache folder
        os.makedirs(cache_folder, exist_ok=True)
        
        try:
            # List objects in S3 folder
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=s3_prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No files found in s3://{self.bucket_name}/{s3_prefix}")
                return folder_path
            
            # Download each file
            for obj in response['Contents']:
                s3_key = obj['Key']
                # Skip folder objects
                if s3_key.endswith('/'):
                    continue
                
                # Get relative path and create local directories
                rel_path = s3_key[len(s3_prefix):]
                local_path = os.path.join(cache_folder, rel_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                
                logger.info(f"Downloading {s3_key} to {local_path}")
                self.s3_client.download_file(self.bucket_name, s3_key, local_path)
            
            logger.info(f"Folder downloaded successfully to {cache_folder}")
            return cache_folder
        except Exception as e:
            logger.error(f"Error downloading folder: {str(e)}")
            # Return original path as fallback
            return folder_path
    
    def is_serverless_env(self):
        """
        Check if running in a serverless environment.
        
        Returns:
            bool: True if running in Lambda, False otherwise
        """
        return 'AWS_LAMBDA_FUNCTION_NAME' in os.environ 