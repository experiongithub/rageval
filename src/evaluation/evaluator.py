from typing import List
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from ..config.evaluation_config import GEVAL_CONFIG, EVALUATION_CRITERIA
from ..models.aws_bedrock import AWSBedrock
from langchain_aws import ChatBedrock

class Evaluator:
    def __init__(self, model: ChatBedrock):
        # Wrap the ChatBedrock model in our custom AWSBedrock class
        self.model = AWSBedrock(model=model)
        
    def create_geval_metric(self) -> GEval:
        """
        Create G-Eval metric with configured settings
        """
        # Create evaluation parameters
        eval_params = [
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT
        ]
        
        return GEval(
            name="response_accuracy_metric",
            criteria=EVALUATION_CRITERIA,
            evaluation_params=eval_params,
            model=self.model,  # Using our custom AWS Bedrock model
            threshold=GEVAL_CONFIG["threshold"],
            strict_mode=GEVAL_CONFIG["strict_mode"],
            async_mode=GEVAL_CONFIG["async_mode"],
            verbose_mode=GEVAL_CONFIG["verbose_mode"]
        )
        
    def evaluate_test_cases(self, test_cases: List[LLMTestCase]) -> dict:
        """
        Evaluate test cases using G-Eval
        
        Args:
            test_cases: List of test cases to evaluate
            
        Returns:
            dict: Evaluation results
        """
        metric = self.create_geval_metric()
        results = evaluate(test_cases=test_cases, metrics=[metric])
        return results