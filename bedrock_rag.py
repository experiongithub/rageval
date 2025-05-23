
"""
Demo script to test if your RAG (Retrieval-Augmented Generation) setup works with AWS Bedrock.

How to run:
    python bedrock_rag.py

You will be prompted to enter questions. Type 'quit' to exit.
Make sure your .env is configured with AWS and knowledge base credentials.
"""

import boto3
import json
from typing import Optional
from dotenv import load_dotenv
import os
from config import ModelConfig, RAGConfig

# Load environment variables from .env
load_dotenv()

# Set up AWS Bedrock clients for retrieval and generation
bedrock_runtime = boto3.client('bedrock-runtime', region_name=os.getenv('AWS_BEDROCK_REGION'))
bedrock_agent = boto3.client('bedrock-agent-runtime', region_name=os.getenv('AWS_BEDROCK_REGION'))

def retrieve_and_generate(knowledge_base_id, model_id, prompt):
    """
    Retrieve relevant context and generate answer using Bedrock.
    """
    try:
        # Retrieve relevant passages from KB
        retrieve_response = bedrock_agent.retrieve(
            knowledgeBaseId=knowledge_base_id,
            retrievalQuery={'text': prompt},
            retrievalConfiguration={
                'vectorSearchConfiguration': {
                    'numberOfResults': RAGConfig.NUMBER_OF_RESULTS
                }
            }
        )
        
        # Collect retrieved passages
        passages = []
        for result in retrieve_response.get('retrievalResults', []):
            text = result.get('content', {}).get('text', '')
            if text:
                passages.append(text)
        
        if not passages:
            # No relevant information found in the knowledge base
            return "No relevant information found in the knowledge base."
        
        # Create a prompt with context
        context = "\n".join(passages)
        enhanced_prompt = RAGConfig.PROMPT_TEMPLATE.format(context=context, question=prompt)
        
        # Debug print enhanced prompt
        print("\n=== Enhanced Prompt ===")
        print(enhanced_prompt)
        print("=====================")
                
        # Generate response using LLaMA3
        llm_response = bedrock_runtime.invoke_model(
            modelId=model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "prompt": enhanced_prompt,
                "max_gen_len": ModelConfig.MAX_GEN_LEN,
                "temperature": ModelConfig.TEMPERATURE,
                "top_p": ModelConfig.TOP_P
            })
        )
        
        response_body = json.loads(llm_response["body"].read())
        
        # Handle different response formats
        if "generation" in response_body:
            return response_body["generation"]
        elif "outputs" in response_body and isinstance(response_body["outputs"], list):
            return response_body["outputs"][0].get("text", "No response generated")
        else:
            print(f"\nDebug - Received response format: {response_body.keys()}")
            return "Unexpected response format."
        
    except Exception as e:
        print(f"\nError in retrieve_and_generate: {str(e)}")
        return None

def main():
    # Get configuration from environment variables
    knowledge_base_id = os.getenv('KNOWLEDGE_BASE_ID')
    model_id = os.getenv('AWS_BEDROCK_MODEL_ID')
    
    if not all([knowledge_base_id, model_id]):
        print("Error: Missing required environment variables. Please check your .env file.")
        return
    
    print("\n=== Bedrock RAG Query System ===")
    print("Type 'quit' to exit")
    print("--------------------------------")
    
    while True:
        user_input = input("\nEnter your question: ")
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        print("\nQuerying knowledge base...")
        response = retrieve_and_generate(knowledge_base_id, model_id, user_input)
        
        if response:
            print("\n=== Response ===")
            print(response)
            print("---------------")

if __name__ == "__main__":
    main()