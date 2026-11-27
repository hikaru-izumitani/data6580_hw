import os
import json

from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
)

from data5580_hw.models.prediction import Prediction


class LLMGateway:

    def init_app(self, app):
        self.PROVIDERS = {
            "groq": {
                "api_key": os.getenv("GROQ_API_KEY"),
                "base_url": "https://api.groq.com/openai/v1",
                "model": "openai/gpt-oss-120b",
            },
            "gemini": {
                "api_key": os.getenv("GEMINI_API_KEY"),
                "base_url": (
                    "https://generativelanguage.googleapis.com/v1beta/openai/"
                ),
                "model": "gemini-3.8-flash",
            },
            "openrouter": {
                "api_key": os.getenv("OPENROUTER_API_KEY"),
                "base_url": "https://openrouter.ai/api/v1",
                "model": os.getenv(
                    "OPENROUTER_MODEL",
                    "google/gemma-4-26b-a4b-it:free"
                ),
            },
        }

        self.LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

        if self.LLM_PROVIDER not in self.PROVIDERS:
            raise ValueError(f"Unsupported provider: {self.LLM_PROVIDER}")

        config = self.PROVIDERS[self.LLM_PROVIDER]

        if not config["api_key"]:
            raise EnvironmentError(
                f"Set the API key for {self.LLM_PROVIDER} before running."
            )

        llm_client = AsyncOpenAI(
            api_key=config["api_key"],
            base_url=config["base_url"],
        )

        self.agent_model = OpenAIChatCompletionsModel(
            model=config["model"],
            openai_client=llm_client,
        )

        print(f"Using {self.LLM_PROVIDER}: {config['model']}")

    @staticmethod
    def _get_super_prompt() -> str:
        return """
        You are an expert Machine Learning Explainability Agent.
    
        Your job is to explain individual machine learning predictions
        using three sources of information:
    
        1. MODEL PREDICTIONS
           - Explain what the model predicted.
           - Explain the confidence or probability if provided.
           - Never confuse probability with certainty.
    
        2. SHAP VALUES
           - Explain which features influenced the prediction.
           - Positive SHAP values increase the model output.
           - Negative SHAP values decrease the model output.
           - Explain the relative importance of the features.
           - Do not confuse feature values with SHAP contributions.
           - Do not claim causation from SHAP values.
    
        3. EMBEDDINGS / SIMILAR OBSERVATIONS
           - Use provided nearest-neighbor observations as context.
           - Identify similarities and differences.
           - Explain whether the prediction is consistent with similar cases.
           - Do not interpret raw embedding dimensions as meaningful
             business features.
           - Do not invent similar observations.
    
        Produce an explanation with the following sections:
    
        ## 1. Prediction Summary
        Describe the prediction in plain English.
    
        ## 2. Key Factors
        Identify the most important positive and negative
        feature contributions using SHAP.
    
        ## 3. Similar Observations
        Explain what the nearest neighbors tell us about this prediction.
    
        ## 4. Potential Risks
        Identify unusual feature values, conflicting signals,
        or uncertainty in the available evidence.
    
        ## 5. Business Interpretation
        Explain what this prediction means to a non-technical
        business user.
    
        ## 6. Technical Interpretation
        Summarize the model output, SHAP contributions,
        and supporting evidence.
    
        IMPORTANT:
        - Only use information supplied in the input.
        - Never invent feature values or explanations.
        - Explicitly identify missing information.
        - Distinguish model evidence from your interpretation.
        - Do not claim that SHAP explains causation.
        - If the evidence is insufficient, say so.
        """

    async def explain_prediction(
            self,
            explanation_agent,
            prediction: dict,
            shap_values: dict,
            similar_observations: list = None,
    ):
        """
        Generate a natural-language explanation of a
        machine learning prediction.

        Args:
            prediction:
                Model output and input feature values.

            shap_values:
                Feature contributions from SHAP.

            similar_observations:
                Nearest neighbors retrieved using embeddings.
        """

        explanation_input = {
            "prediction": prediction,
            "shap_values": shap_values,
            "similar_observations": similar_observations or [],
        }

        result = await Runner.run(
            explanation_agent,
            input=json.dumps(explanation_input, indent=2),
        )

        return result.final_output

    def generate_explanation(self, prediction: Prediction) -> Prediction:

        explanation_agent = Agent(
            name="ML Explainability Agent",

            model=self.agent_model,

            instructions=self._get_super_prompt(),
        )

        explanation = self.explain_prediction(
            prediction=prediction,
            shap_values=shap_values,
            similar_observations=similar_observations,
        )

        print(explanation)



        return prediction


llm_gateway = LLMGateway()
