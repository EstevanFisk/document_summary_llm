from providers import GeminiClient, OpenAIClient
from typing import Dict, List
from langchain_core.documents import Document
from config.settings import settings
import json


class ResearchAgent:
    def __init__(self):
        """
        Initialize the research agent with Gemini (Primary) and OpenAI (Fallback).
        """
        print("Initializing ResearchAgent...")
        
        # 1. Initialize Primary (Gemini)
        try:
            # Note: This uses the default config from your GeminiClient. 
            # If you strictly need temperature=0.3, you might need to update GeminiClient to accept config.
            self.primary_client = GeminiClient(api_key=settings.GOOGLE_API_KEY)
            print(" - Primary (Gemini) initialized.")
        except Exception as e:
            print(f"Warning: Could not init Gemini: {e}")
            self.primary_client = None

        # 2. Initialize Secondary (OpenAI)
        try:
            self.secondary_client = OpenAIClient(api_key=settings.OPENAI_API_KEY)
            print(" - Secondary (OpenAI) initialized.")
        except Exception as e:
            print(f"Warning: Could not init OpenAI: {e}")
            self.secondary_client = None

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        """
        prompt = f"""
        You are an AI assistant designed to provide precise and factual answers based on the given context.

        **Instructions:**
        - Answer the following question using only the provided context.
        - Be clear, concise, and factual.
        - Return as much information as you can get from the context.
        
        **Question:** {question}
        **Context:**
        {context}

        **Provide your answer below:**
        """
        return prompt
    

    def _get_llm_response(self, prompt: str) -> str:
        """
        Refined 2026 fallback logic with rate-limit awareness.
        """
        constraints = {
            "max_tokens": 4000, # Keep responses concise to save tokens
            "temperature": 0.3
        }

        # 1. ALWAYS try Gemini first (Better free tier limits)
        if self.primary_client:
            try:
                print("Attempting research with Gemini...")
                return self.primary_client.generate(prompt, **constraints)
            except Exception as e:
                print(f"Gemini (Primary) unavailable: {e}")
        
        # 2. Try OpenAI second (The 'Safety Net')
        if self.secondary_client:
            try:
                print("Attempting fallback with OpenAI...")
                return self.secondary_client.generate(prompt, **constraints)
            except Exception as e:
                # If OpenAI also fails with a 429, we give a specific error
                if "429" in str(e):
                    raise RuntimeError("Rate limit reached. Please wait 60 seconds.") from e
                print(f"OpenAI (Fallback) failed: {e}")

        raise RuntimeError("No AI models are currently responding. Check your API keys.")
    

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """
        Generate an initial answer using the provided documents.
        """
        print(f"ResearchAgent.generate called for: '{question}'")

        # 1. Combine the document contents into one context string
        context = "\n\n".join([doc.page_content for doc in documents])
        
        # 2. Create the prompt
        prompt = self.generate_prompt(question, context)

        # 3. Call the LLM using our helper (Gemini -> OpenAI fallback)
        try:
            print("Sending prompt to the model...")
            # We call our helper which returns a raw string
            raw_answer = self._get_llm_response(prompt)
            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to generate answer due to a model error.") from e

        # 4. Sanitize and return
        draft_answer = self.sanitize_response(raw_answer) if raw_answer else "I cannot answer this question."

        return {
            "draft_answer": draft_answer,
            "context_used": context
        }