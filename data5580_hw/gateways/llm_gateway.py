import os
import json


class LLMGateway:
    def init_app(self,app):
        ...
    def generate_explanation(self, prediction_id: str) -> str:
        ...
    @staticmethod
    def _get_super_prompt() -> str:
        ...
    
    @staticmethod
    async def explain_prediction(prediction_id: str) -> str:
        ...

    def generate_explanation(self, prediction_id: str) -> str:
        explain_agent = Agent(
            name,
            model,
            instructions
        )
        