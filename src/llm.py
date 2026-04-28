import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

MODEL_NAME = "gemini-2.0-flash-lite"


def load_api_key() -> str:
    """
    Load and validate the Gemini API key from the environment.

    Checks GOOGLE_API_KEY first, then GEMINI_API_KEY as a fallback,
    and ensures the SDK-preferred variable is always set.

    Returns:
        The API key string.

    Raises:
        EnvironmentError: If neither key is present.
    """
    key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not key:
        raise EnvironmentError(
            "Neither GOOGLE_API_KEY nor GEMINI_API_KEY is set."
        )
    os.environ.setdefault("GOOGLE_API_KEY", key)
    return key


def build_llm(temperature: float = 0.3) -> BaseChatModel:
    """
    Instantiate the Gemini chat model for RAG answer generation.

    Args:
        temperature: Sampling temperature. Lower = more factual.

    Returns:
        A configured BaseChatModel instance.
    """
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=temperature,
        max_retries=3,
    )