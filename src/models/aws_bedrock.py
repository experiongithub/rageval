from deepeval.models import DeepEvalBaseLLM
from langchain_aws import ChatBedrock
from typing import Any
from ..config.evaluation_config import EVALUATOR_MODEL_CONFIG

class AWSBedrock(DeepEvalBaseLLM):
    def __init__(self, model: ChatBedrock):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return EVALUATOR_MODEL_CONFIG["model_name"]