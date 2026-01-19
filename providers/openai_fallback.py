# providers/openai_fallback.py
from openai import OpenAI
from .base import LLMClient
# pip install openai

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("OpenAI API Key is required for OpenAIClient")
            
        self.client = OpenAI(api_key=api_key)
        self.model_name = "gpt-4o-mini"

    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> str:
        """
        Generates text using OpenAI, matching the signature of GeminiClient for agnostic use.
        """
        try:
            # We apply the strict limits HERE, for every specific request
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
            
        except Exception as e:
            raise RuntimeError(f"OpenAI Error: {e}")