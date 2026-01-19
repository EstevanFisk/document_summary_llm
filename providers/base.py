# providers/base.py
from abc import ABC, abstractmethod

class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, max_tokens: int = 200, temperature: float = 0.0) -> str:
        """
        Generates text from the LLM.
        
        Args:
            prompt: The input text.
            max_tokens: Hard limit on output length (default 200).
            temperature: Randomness control (default 0.0).
        """
        pass