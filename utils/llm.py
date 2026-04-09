import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.0, max_tokens: int = 1000) -> ChatOpenAI:
    """
    Single place to configure the LLM. Swap model here if needed.
    temperature=0 keeps extraction deterministic.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    return ChatOpenAI(
        model="gpt-4o",
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
    )
