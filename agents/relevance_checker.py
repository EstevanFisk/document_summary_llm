from providers import GeminiClient, OpenAIClient
from config.settings import settings
import re
import logging

logger = logging.getLogger(__name__)

class RelevanceChecker:
    def __init__(self):
        """
        Initialize the relevance checker with Gemini (Primary) and OpenAI (Fallback).
        Using strict settings (Temp=0) for classification consistency.
        """
        # 1. Initialize Primary (Gemini)
        try:
            self.primary_client = GeminiClient(api_key=settings.GOOGLE_API_KEY)
            logger.info("RelevanceChecker: Primary (Gemini) initialized.")
        except Exception as e:
            logger.warning(f"RelevanceChecker: Could not init Gemini: {e}")
            self.primary_client = None

        # 2. Initialize Secondary (OpenAI)
        try:
            self.secondary_client = OpenAIClient(api_key=settings.OPENAI_API_KEY)
            logger.info("RelevanceChecker: Secondary (OpenAI) initialized.")
        except Exception as e:
            logger.warning(f"RelevanceChecker: Could not init OpenAI: {e}")
            self.secondary_client = None

    def _get_llm_response(self, prompt: str) -> str:
        """
        INTERNAL HELPER: Handles the fallback logic.
        Returns the raw string from whichever model works.
        """
        # 100 tokens is a safe buffer for 'Thinking' models to reach the final word
        constraints = {
            "max_tokens": 1000,
            "temperature": 0.0
        }

        # 1. Try Primary (Gemini)
        if self.primary_client:
            try:
                print("--- RelevanceChecker: Attempting Gemini ---")
                return self.primary_client.generate(prompt, **constraints)
            except Exception as e:
                print(f"--- RelevanceChecker: Gemini Failed: {e} ---")
                logger.warning(f"Gemini failed ({e}). Switching to fallback...")
        else:
            print("--- RelevanceChecker: GeminiClient is NONE (Initialization failed) ---")
        
        # 2. Try Secondary (OpenAI)
        if self.secondary_client:
            print("--- RelevanceChecker: Attempting OpenAI Fallback ---")
            try:
                return self.secondary_client.generate(prompt, **constraints)
            except Exception as e:
                print(f"--- RelevanceChecker: OpenAI Failed: {e} ---")
                logger.error(f"OpenAI failed: {e}")
        
        # 3. Fail
        raise RuntimeError("All models failed to generate a response.")

    def check(self, question: str, retriever, k=3) -> str:
        """
        1. Retrieve top-k document chunks.
        2. Combine them into a single string.
        3. Classify relevance using Fuzzy Matching to handle token cutoffs.
        """
        logger.debug(f"RelevanceChecker.check called with question='{question}' and k={k}")

        # Retrieve doc chunks from the ensemble retriever
        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned from retriever.invoke(). Classifying as NO_MATCH.")
            return "NO_MATCH"

        # Combine the top k chunk texts into one string
        document_content = "\n\n".join(doc.page_content for doc in top_docs[:k])

        # Create the classification prompt
        prompt = f"""
        You are an AI relevance checker. 
        Classify how well the document content addresses the user's question.

        **Instructions:**
        - Respond with ONLY one label: CAN_ANSWER, PARTIAL, or NO_MATCH.
        - Do not provide any explanation.

        **Labels:**
        1) "CAN_ANSWER": The passages contain enough information to fully answer.
        2) "PARTIAL": The passages discuss the topic but lack some details.
        3) "NO_MATCH": The passages do not mention the topic at all.

        **Question:** {question}
        **Passages:** {document_content}

        **Respond ONLY with one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH**
        """

        try:
            # Get the raw string response
            llm_response_text = self._get_llm_response(prompt)
        except RuntimeError:
            logger.error("All models failed. Defaulting to NO_MATCH.")
            return "NO_MATCH"

        # Clean the response for processing
        llm_response = llm_response_text.strip().upper()
        print(f"Checker raw response: {llm_response}")

        # --- RESILIENT FUZZY MATCHING ---
        # Instead of '==' we use 'in' to handle cases where the model is cut off
        # or adds extra reasoning (e.g., 'Label: PARTIAL' or 'PART' due to token limits).
        if "CAN_ANSWER" in llm_response:
            classification = "CAN_ANSWER"
        elif "PART" in llm_response: 
            # This catches "PART", "PARTIAL", "PARTIALLY", or "PART..."
            classification = "PARTIAL"
        elif "NO_MATCH" in llm_response:
            classification = "NO_MATCH"
        else:
            # If the model returns something completely unexpected, default to NO_MATCH
            logger.debug(f"Unexpected LLM output: {llm_response}. Forcing 'NO_MATCH'.")
            classification = "NO_MATCH"

        print(f"Final Classification: {classification}")
        return classification