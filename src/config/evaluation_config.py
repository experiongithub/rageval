# Evaluation System Configuration
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Evaluator Model Configuration
EVALUATOR_MODEL_CONFIG = {
    "region_name": os.getenv("AWS_EVALUATOR_REGION"),  # Separate region for evaluator
    "model_id": os.getenv("AWS_EVALUATOR_MODEL_ID"),   # Potentially different/superior model
    "model_name": os.getenv("AWS_EVALUATOR_MODEL_NAME", "Custom AWS Bedrock Model"),  # Display name
    "model_kwargs": {
        "temperature": 0.7,
        "top_p": 0.8,
        "max_gen_len": 2048
    }
}

# Validate required environment variables
if not EVALUATOR_MODEL_CONFIG["region_name"] or not EVALUATOR_MODEL_CONFIG["model_id"]:
    raise ValueError("AWS_EVALUATOR_REGION and AWS_EVALUATOR_MODEL_ID must be set in environment variables")

# Data Paths for Evaluation
DATA_PATHS = {
    "golden_test_cases": "synthetic_data/goldens.json"
}

# G-Eval Configuration
GEVAL_CONFIG = {
    "threshold": 0.7,  # passing threshold
    "strict_mode": False,  # binary scoring
    "async_mode": True,  # concurrent execution
    "verbose_mode": False  # debug output
}

# Evaluation Criteria
EVALUATION_CRITERIA = """
Evaluate the response for:
1. Factual accuracy based on the provided context
2. Completeness of information
3. Relevance to the query
4. Coherence and clarity
"""

def create_evaluator_model() -> 'ChatBedrock':
    """Create and return a configured AWS Bedrock model instance for evaluation."""
    from langchain_aws import ChatBedrock
    
    return ChatBedrock(
        model_id=EVALUATOR_MODEL_CONFIG["model_id"],
        region_name=EVALUATOR_MODEL_CONFIG["region_name"],
        model_kwargs=EVALUATOR_MODEL_CONFIG["model_kwargs"]
    )