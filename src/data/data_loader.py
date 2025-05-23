from typing import List, Optional
from pathlib import Path
import json
from ..config.evaluation_config import DATA_PATHS

class GoldenTestCase:
    def __init__(
        self,
        input: str,
        expected_output: str,
        context: List[str],
        source_file: Optional[str] = None
    ):
        self.input = input
        self.expected_output = expected_output
        self.context = context
        self.source_file = source_file

class DataLoader:
    def __init__(self, golden_path: str):
        self.golden_path = Path(golden_path)
        
    def load_golden_testcases(self) -> List[GoldenTestCase]:
        """
        Load golden test cases from the JSON file
        """
        if not self.golden_path.exists():
            raise FileNotFoundError(f"Golden test cases file not found: {self.golden_path}")
            
        with open(self.golden_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        test_cases = []
        for item in data:
            test_case = GoldenTestCase(
                input=item['input'],
                expected_output=item['expected_output'],
                context=item['context'],
                source_file=item.get('source_file')
            )
            test_cases.append(test_case)
            
        return test_cases

def get_data_loader(golden_path: str = DATA_PATHS["golden_test_cases"]) -> DataLoader:
    """
    Factory function to create a DataLoader instance
    """
    return DataLoader(golden_path)