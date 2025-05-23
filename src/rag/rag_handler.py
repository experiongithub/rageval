import boto3
import os
from typing import List, Optional
from dotenv import load_dotenv
from .knowledge_base import KnowledgeBase
from ..config.rag_config import BEDROCK_CONFIG, KNOWLEDGE_BASE_CONFIG, create_bedrock_model

# Load environment variables
load_dotenv()

class RAGHandler:
    def __init__(self, knowledge_base_id: str):
        """
        Initialize RAG handler
        
        Args:
            knowledge_base_id: ID of the AWS Knowledge Base
        """
        self.knowledge_base_id = knowledge_base_id
        self.knowledge_base = KnowledgeBase(knowledge_base_id)
        
        # Initialize Bedrock clients
        self.bedrock_runtime = boto3.client(
            'bedrock-runtime', 
            region_name=BEDROCK_CONFIG["region_name"]
        )
        self.bedrock_agent = boto3.client(
            'bedrock-agent-runtime', 
            region_name=BEDROCK_CONFIG["region_name"]
        )
        
        # Store configurations
        self.config = BEDROCK_CONFIG
        self.model = create_bedrock_model()
        
    def get_rag_response(self, query: str) -> str:
        """
        Get RAG response by:
        1. Retrieving context from knowledge center using Bedrock
        2. Getting response from RAG using retrieved context
        
        Args:
            query: Input query
            
        Returns:
            str: Generated response from RAG
        """
        # Step 1: Retrieve context from knowledge center using Bedrock
        try:
            retrieve_response = self.bedrock_agent.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={
                    'text': query
                },
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': KNOWLEDGE_BASE_CONFIG["num_results"]
                    }
                }
            )
            
            # Extract passages from response
            retrieved_contexts = []
            for result in retrieve_response.get('retrievalResults', []):
                text = result.get('content', {}).get('text', '')
                if text:
                    retrieved_contexts.append(text)
                    
            # Format retrieved context
            formatted_context = self.knowledge_base.format_context(retrieved_contexts)
            
        except Exception as e:
            print(f"Error in context retrieval: {str(e)}")
            return "Error: Failed to retrieve context"

        # Step 2: Create prompt with retrieved context
        prompt = f"""Based on the following context, please answer the question.

Context:
{formatted_context}

Question:
{query}

Answer:"""
        
        # Step 3: Get response from RAG model
        try:
            response = self.model.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error in RAG response generation: {str(e)}")
            return "Error: Failed to generate response"

    def process_test_case(self, input_query: str) -> str:
        """
        Process a single test case to get RAG response
        
        Args:
            input_query: Input question
            
        Returns:
            str: RAG response
        """
        return self.get_rag_response(input_query)