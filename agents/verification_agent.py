import json  # Import for JSON serialization
from providers import GeminiClient, OpenAIClient
from typing import Dict, List
from langchain_core.documents import Document
from config.settings import settings


class VerificationAgent:
    def __init__(self):
        """
        Initialize the verification agent with Gemini (Primary) and OpenAI (Fallback).
        """
        print("Initializing VerificationAgent...")
        
        # 1. Initialize Primary (Gemini)
        try:
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

    def generate_prompt(self, answer: str, context: str) -> str:
        """
        Generate a structured prompt for the LLM to verify the answer against the context.
        """
        prompt = f"""
        You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

        **Instructions:**
        - Verify the following answer against the provided context.
        - Check for:
        1. Direct/indirect factual support (YES/NO)
        2. Unsupported claims (list any if present)
        3. Contradictions (list any if present)
        4. Relevance to the question (YES/NO)
        - Provide additional details or explanations where relevant.
        - Respond in the exact format specified below without adding any unrelated information.

        **Format:**
        Supported: YES/NO
        Unsupported Claims: [item1, item2, ...]
        Contradictions: [item1, item2, ...]
        Relevant: YES/NO
        Additional Details: [Any extra information or explanations]

        **Answer:** {answer}
        **Context:**
        {context}

        **Respond ONLY with the above format.**
        """
        return prompt

    def parse_verification_response(self, response_text: str) -> Dict:
        """
        Parse the LLM's verification response into a structured dictionary.
        """
        try:
            lines = response_text.split('\n')
            verification = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    key = key.strip().capitalize()
                    value = value.strip()
                    if key in {"Supported", "Unsupported claims", "Contradictions", "Relevant", "Additional details"}:
                        if key in {"Unsupported claims", "Contradictions"}:
                            # Convert string list to actual list
                            if value.startswith('[') and value.endswith(']'):
                                items = value[1:-1].split(',')
                                # Remove any surrounding quotes and whitespace
                                items = [item.strip().strip('"').strip("'") for item in items if item.strip()]
                                verification[key] = items
                            else:
                                verification[key] = []
                        elif key == "Additional details":
                            verification[key] = value
                        else:
                            verification[key] = value.upper()
            # Ensure all keys are present
            for key in ["Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"]:
                if key not in verification:
                    if key in {"Unsupported Claims", "Contradictions"}:
                        verification[key] = []
                    elif key == "Additional Details":
                        verification[key] = ""
                    else:
                        verification[key] = "NO"

            return verification
        except Exception as e:
            print(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        """
        Format the verification report dictionary into a readable paragraph.
        """
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        report = f"**Supported:** {supported}\n"
        if unsupported_claims:
            report += f"**Unsupported Claims:** {', '.join(unsupported_claims)}\n"
        else:
            report += f"**Unsupported Claims:** None\n"

        if contradictions:
            report += f"**Contradictions:** {', '.join(contradictions)}\n"
        else:
            report += f"**Contradictions:** None\n"

        report += f"**Relevant:** {relevant}\n"

        if additional_details:
            report += f"**Additional Details:** {additional_details}\n"
        else:
            report += f"**Additional Details:** None\n"

        return report
    

    def _get_llm_response(self, prompt: str) -> str:
        """
        Refined 2026 fallback logic with rate-limit awareness.
        """
        constraints = {
            "max_tokens": 3000, # Keep responses concise to save tokens
            "temperature": 0.0
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
    

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """
        Verify the answer against the provided documents.
        """
        print(f"VerificationAgent.check called with answer and {len(documents)} documents.")

        # Combine all document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents])
        print(f"Combined context length: {len(context)} characters.")

        # Create a prompt for the LLM to verify the answer
        prompt = self.generate_prompt(answer, context)
        print("Prompt created for the LLM.")

        # 1. FIX: Call the helper instead of self.model.chat
        try:
            print("Sending verification prompt to the model...")
            llm_response = self._get_llm_response(prompt) # Returns raw string
            print("LLM response received.")
        except Exception as e:
            print(f"Error during model inference: {e}")
            raise RuntimeError("Failed to verify answer due to a model error.") from e

        # 2. Extract and process the LLM's response
        # Since _get_llm_response already gives us the content string, we just sanitize it
        sanitized_response = self.sanitize_response(llm_response) if llm_response else ""

        if not sanitized_response:
            print("LLM returned an empty response.")
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model."
            }
        else:
            # 3. Parse the response using your existing parser
            verification_report = self.parse_verification_response(sanitized_response)
            if verification_report is None:
                print("LLM did not respond with expected format. Using default.")
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response."
                }

        # 4. Format the final report for the UI
        verification_report_formatted = self.format_verification_report(verification_report)
        print(f"Verification report completed.")

        return {
            "verification_report": verification_report_formatted,
            "context_used": context
        }
