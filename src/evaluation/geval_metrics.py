from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.models.base_model import DeepEvalBaseLLM

def create_accuracy_metric(model: DeepEvalBaseLLM) -> GEval:
    """
    Create a GEval metric for evaluating RAG responses.
    
    Args:
        model: The LLM model to use for evaluation
        
    Returns:
        GEval: Configured evaluation metric
    """
    return GEval(
        name="response_accuracy_metric",
        criteria="""Evaluate the response for:
1. Factual Accuracy: Information provided matches the source material
2. Completeness: All key points from the question are addressed
3. Clarity: Response is well-structured and easy to understand""",
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.CONTEXT
        ],
        model=model
    )