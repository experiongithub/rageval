"""Configuration for RAG system"""

class ModelConfig:
    """LLaMA3 model configuration"""
    MAX_GEN_LEN = 300
    TEMPERATURE = 0.7
    TOP_P = 0.8

class RAGConfig:
    """RAG system configuration"""
    NUMBER_OF_RESULTS = 3
    
    # Prompt template
    PROMPT_TEMPLATE = """Based on the following context, please answer the question.

Context:
{context}

Question: {question}

Answer:"""