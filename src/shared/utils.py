"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
    get_llm_provider: Resolve the LLM provider from the LLM_PROVIDER env var.
    default_chat_model_name: Default 'provider/model' name for the active LLM provider.
"""

import os
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel


def _format_doc(doc: Document) -> str:
    """Format a single document as XML.

    Args:
        doc (Document): The document to format.

    Returns:
        str: The formatted document as an XML string.
    """
    metadata = doc.metadata or {}
    meta = "".join(f" {k}={v!r}" for k, v in metadata.items())
    if meta:
        meta = f" {meta}"

    return f"<document{meta}>\n{doc.page_content}\n</document>"


def format_docs(docs: Optional[list[Document]]) -> str:
    """Format a list of documents as XML.

    This function takes a list of Document objects and formats them into a single XML string.

    Args:
        docs (Optional[list[Document]]): A list of Document objects to format, or None.

    Returns:
        str: A string containing the formatted documents in XML format.

    Examples:
        >>> docs = [Document(page_content="Hello"), Document(page_content="World")]
        >>> print(format_docs(docs))
        <documents>
        <document>
        Hello
        </document>
        <document>
        World
        </document>
        </documents>

        >>> print(format_docs(None))
        <documents></documents>
    """
    if not docs:
        return "<documents></documents>"
    formatted = "\n".join(_format_doc(doc) for doc in docs)
    return f"""<documents>
{formatted}
</documents>"""


SUPPORTED_LLM_PROVIDERS = ("gemini", "bedrock")


def get_llm_provider() -> str:
    """Return the LLM provider selected by the LLM_PROVIDER env var (default 'gemini')."""
    provider = os.getenv("LLM_PROVIDER", "gemini").strip().lower() or "gemini"
    if provider not in SUPPORTED_LLM_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER: {provider!r}. "
            f"Expected one of: {', '.join(SUPPORTED_LLM_PROVIDERS)}"
        )
    return provider


def default_chat_model_name() -> str:
    """Return the default 'provider/model' name for the active LLM provider."""
    if get_llm_provider() == "bedrock":
        model = os.getenv(
            "BEDROCK_MODEL", "us.anthropic.claude-haiku-4-5-20251001-v1:0"
        )
        return f"bedrock_converse/{model}"
    return f"google_genai/{os.getenv('GEMINI_MODEL', 'gemini-3.5-flash-lite')}"


def load_chat_model(fully_specified_name: str) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name (str): String in the format 'provider/model'.
    """
    if "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        provider = ""
        model = fully_specified_name
    if provider in ("bedrock", "bedrock_converse"):
        return init_chat_model(
            model,
            model_provider="bedrock_converse",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
        )
    return init_chat_model(model, model_provider=provider)
