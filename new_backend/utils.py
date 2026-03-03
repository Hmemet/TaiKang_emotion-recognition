import os
os.environ['HF_HOME'] = 'C:/hf_cache'  
os.environ['TRANSFORMERS_CACHE'] = 'C:/hf_cache'

import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Backend-Utils")

def clean_text(text: str) -> str:
   
    if not text:
        return ""
    
    text = text.strip()
    
    text = re.sub(r'\s+', ' ', text)
    
    
    return text

def format_api_response(status: str, data: dict = None, message: str = ""):
    """
    统一 API 返回格式
    """
    return {
        "status": status,
        "message": message,
        "data": data,
        "timestamp": logging.time.time()
    }