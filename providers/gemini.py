# providers/gemini.py
import google.generativeai as genai
from .base import LLMClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class GeminiClient(LLMClient):
    def __init__(self, api_key: str):
        """
        Initialize the Gemini client with the 2026 standard SDK.
        """
        if not api_key:
            raise ValueError("Google API Key is required for GeminiClient")
            
        genai.configure(api_key=api_key)
        # Using the stable 2026 flash model for speed and long context
        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def generate(self, prompt: str, max_tokens: int = 1500, temperature: float = 0.0) -> str:
        """
        Generates text with robust handling for token cutoffs and safety blocks.
        """
        try:
            config = genai.types.GenerationConfig(
                max_output_tokens=max_tokens,
                temperature=temperature
            )
            
            response = self.model.generate_content(
                prompt, 
                generation_config=config,
                safety_settings={k: "BLOCK_NONE" for k in [
                    "HARM_CATEGORY_HARASSMENT", 
                    "HARM_CATEGORY_HATE_SPEECH", 
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", 
                    "HARM_CATEGORY_DANGEROUS_CONTENT"
                ]}
            )

            # --- ROBUST CANDIDATE CHECK ---
            # 1. Verify that the response actually contains data
            if not hasattr(response, 'candidates') or not response.candidates:
                print("--- Gemini Warning: No candidates returned (Possible Safety/Quota block) ---")
                return ""

            candidate = response.candidates[0]

            # 2. Handle 'Finish Reason 2' (Max Tokens) or other partial responses
            # We manually extract the text parts to avoid the response.text exception
            try:
                if candidate.content and candidate.content.parts:
                    text_content = candidate.content.parts[0].text
                    if candidate.finish_reason == 2:
                        print("--- DEBUG: Gemini reached max tokens. Returning partial text. ---")
                    return text_content
            except (AttributeError, IndexError):
                pass

            # 3. Final Fallback to standard accessor
            try:
                return response.text if response.text else ""
            except (ValueError, AttributeError):
                return ""
            
        except Exception as e:
            # Combined error handler to prevent app crashes while alerting the workflow
            print(f"--- Fatal Gemini Error: {e} ---")
            raise RuntimeError(f"Gemini Error: {e}")

class GeminiEmbeddings:
    @staticmethod
    def get_embeddings(api_key: str):
        """
        Returns a LangChain-compatible Google embedding object.
        """
        if not api_key:
            raise ValueError("Google API Key is required for Embeddings")
        
        return GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=api_key
        )