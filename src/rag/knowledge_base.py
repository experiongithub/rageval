from typing import List
from ..config.rag_config import BEDROCK_CONFIG
import boto3

class KnowledgeBase:
    def __init__(self, knowledge_base_id: str):
        self.knowledge_base_id = knowledge_base_id
        self.bedrock_agent = boto3.client(
            'bedrock-agent-runtime',
            region_name=BEDROCK_CONFIG["region_name"]
        )
        
    def retrieve_context(self, prompt: str, num_results: int = 3) -> List[str]:
        """
        Retrieve relevant passages using Bedrock Agent Runtime
        
        Args:
            prompt: The input query
            num_results: Number of passages to retrieve
            
        Returns:
            List[str]: Retrieved passages
        """
        try:
            retrieve_response = self.bedrock_agent.retrieve(
                knowledgeBaseId=self.knowledge_base_id,
                retrievalQuery={
                    'text': prompt
                },
                retrievalConfiguration={
                    'vectorSearchConfiguration': {
                        'numberOfResults': num_results
                    }
                }
            )
            
            # Extract passages from response
            passages = []
            for result in retrieve_response.get('retrievalResults', []):
                text = result.get('content', {}).get('text', '')
                if text:
                    passages.append(text)
                    
            return passages
        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            return []

    def format_context(self, contexts: List[str]) -> str:
        """
        Format multiple context pieces into a single string
        """
        return "\n\n".join(contexts)