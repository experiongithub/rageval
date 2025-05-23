# RAG System Configuration
import os
from dotenv import load_dotenv
from langchain_aws import ChatBedrock

# Load environment variables
load_dotenv()

# AWS Bedrock Model Configuration
BEDROCK_CONFIG = {
    "region_name": os.getenv("AWS_BEDROCK_REGION"),  # No default - must be set in .env
    "model_id": os.getenv("AWS_BEDROCK_MODEL_ID"),   # No default - must be set in .env
    "model_kwargs": {
        "temperature": 0.7,
        "top_p": 0.8,
        "max_gen_len": 2048
    }
}

# Validate required environment variables
if not BEDROCK_CONFIG["region_name"] or not BEDROCK_CONFIG["model_id"]:
    raise ValueError("AWS_BEDROCK_REGION and AWS_BEDROCK_MODEL_ID must be set in environment variables")

# Knowledge Base Configuration
KNOWLEDGE_BASE_CONFIG = {
    "knowledge_base_id": os.getenv("KNOWLEDGE_BASE_ID"),
    "num_results": 3
}

def create_bedrock_model() -> ChatBedrock:
    """Create and return a configured AWS Bedrock model instance."""
    return ChatBedrock(**BEDROCK_CONFIG)