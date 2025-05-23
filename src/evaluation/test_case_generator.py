from typing import List
from ..data.data_loader import DataLoader
from ..rag.rag_handler import RAGHandler
from deepeval.test_case import LLMTestCase
import json

class TestCaseGenerator:
    def __init__(self, rag_handler: RAGHandler, data_loader: DataLoader):
        self.rag_handler = rag_handler
        self.data_loader = data_loader
        
    def generate_test_cases(self) -> List[LLMTestCase]:
        """
        Generate test cases from golden examples:
        1. Load golden test cases using DataLoader
        2. For each golden:
           - Get RAG response for input
           - Create LLMTestCase using golden data and RAG response
        
        Returns:
            List[LLMTestCase]: List of test cases ready for G-Eval
        """
        # Load golden test cases using DataLoader
        golden_cases = self.data_loader.load_golden_testcases()
        test_cases = []
        
        for item in golden_cases:
            # Get RAG response for input
            rag_response = self.rag_handler.get_rag_response(
                query=item.input
            )
            
            # Create LLMTestCase using golden data and RAG response
            test_case = LLMTestCase(
                input=item.input,              # From golden
                expected_output=item.expected_output,  # From golden
                context=item.context,          # From golden
                actual_output=rag_response     # From RAG
            )
            test_cases.append(test_case)
            
        return test_cases