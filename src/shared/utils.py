"""Shared utility functions used in the project.

Functions:
    format_docs: Convert documents to an xml-formatted string.
    load_chat_model: Load a chat model from a model name.
"""

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


def load_chat_model(
    fully_specified_name: str, *, provider: Optional[str] = None
) -> BaseChatModel:
    """Load a chat model from a fully specified name.

    Args:
        fully_specified_name: Either a bare model id (e.g. ``"gemini-2.5-flash-lite"``)
            or a ``"provider/model"`` string. If ``provider`` is given, it
            overrides any provider prefix.
        provider: Optional provider name (e.g. ``"google_genai"`` or
            ``"bedrock_converse"``). When set, ``fully_specified_name`` is
            treated as the model id only.

    Notes:
        For ``bedrock_converse``, ``boto3`` automatically picks up
        ``AWS_BEARER_TOKEN_BEDROCK`` (and standard AWS credential env vars)
        from the environment.
    """
    if provider is None and "/" in fully_specified_name:
        provider, model = fully_specified_name.split("/", maxsplit=1)
    else:
        model = fully_specified_name
    
    if provider == "bedrock_converse":
        # Bedrock models require the provider to be specified as a parameter to the model loader, and won't work if the provider is included in the model name.
        return init_chat_model(model, model_provider=provider, region_name="us-east-1")
    return init_chat_model(model, model_provider=provider or None)
