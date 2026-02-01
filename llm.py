"""
LLM Module using direct REST API calls with requests to OpenRouter Mistral-7B-Instruct.
Generates grounded text responses based on provided context.
"""

import logging
from typing import List, Dict
import os
import requests
import json

from dotenv import load_dotenv

import config
import dotenv
load_dotenv()

logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)


class MistralInstructModel:
    """
    Language model wrapper that calls the OpenRouter REST API directly with requests.
    """

    def __init__(
            self,
            api_key: str = None,
            max_tokens: int = 512,
            temperature: float = config.LLM_TEMPERATURE,
            top_p: float = config.LLM_TOP_P,
            site_url: str = None,
            site_name: str = None
    ):
        """
        Initialize with API key and generation parameters.

        Args:
            api_key: OpenRouter API key (defaults to environment variable)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            site_url: Optional HTTP Referer header value for site ranking
            site_name: Optional X-Title header value for site ranking
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be set in environment or passed explicitly")

        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model = "mistralai/mistral-7b-instruct"
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.site_url = site_url
        self.site_name = site_name

    def generate_response(self, query: str, context_chunks: List[str]) -> Dict[str, any]:
        """
        Generate a response from the API given the query and context chunks.

        Args:
            query: The user question
            context_chunks: List of retrieved context strings

        Returns:
            Dictionary with 'response' text, 'sources_used', and optionally 'context'
        """
        if not query or not query.strip():
            return {"response": "Please provide a valid question.", "sources_used": 0}

        context = self._prepare_context(context_chunks)

        prompt = (
            "<s>[INST] "
            "You are a professional product support assistant. "
            "Answer clearly and factually using ONLY the information provided in the context. "
            "Do NOT use emojis, metaphors, marketing language, markdown, bullet styling, or external examples. "
            "If the answer is not present in the context, say that the information is not available. "
            f"\n\nContext:\n{context}\n\nQuestion:\n{query}\n\nAnswer:[/INST]"
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.site_url or "http://localhost",
            "X-Title": self.site_name or "VoiceAssist-Pro"
        }
        if self.site_url:
            headers["HTTP-Referer"] = self.site_url
        if self.site_name:
            headers["X-Title"] = self.site_name

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        try:
            logger.info("Sending request to OpenRouter Mistral-7B-Instruct API...")
            response = requests.post(self.api_url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            data = response.json()

            answer = data["choices"][0]["message"]["content"].strip()
            logger.info("Response generated successfully")

            return {"response": answer, "sources_used": len(context_chunks), "context": context_chunks}

        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error: {http_err} - Response content: {response.text}")
            return {"response": f"HTTP error contacting API: {http_err}", "sources_used": 0}
        except Exception as e:
            logger.error(f"API generation failed: {e}")
            return {"response": f"Failed to generate response: {e}", "sources_used": 0}

    @staticmethod
    def _prepare_context(chunks: List[str], max_context_length: int = 2000) -> str:
        if not chunks:
            return "No relevant information available."
        context_parts = []
        current_length = 0
        for i, chunk in enumerate(chunks, 1):
            chunk_text = f"[Source {i}]: {chunk}"
            length = len(chunk_text)
            if current_length + length > max_context_length:
                break
            context_parts.append(chunk_text)
            current_length += length
        return "\n\n".join(context_parts)


# Example usage
if __name__ == "__main__":
    model = MistralInstructModel(site_url="https://your-site-url.com", site_name="Your Site")
    query = "What is machine learning?"
    context = [
        "Machine learning is a branch of artificial intelligence that focuses on building systems that can learn from data."
    ]
    result = model.generate_response(query, context)
    print("Generated response:")
    print(result["response"])
