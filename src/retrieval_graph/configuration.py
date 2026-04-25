"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from retrieval_graph import prompts
from shared.configuration import BaseConfiguration


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    # models

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="bedrock_converse/google.gemma-3-27b-it",
        metadata={
            "description": "Model used for query generation. Format: 'provider/model-id'. Examples: 'bedrock_converse/us.anthropic.claude-3-5-haiku-20241022-v1:0', 'bedrock_converse/us.amazon.nova-lite-v1:0', 'google_genai/gemini-2.5-flash-lite'."
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default="bedrock_converse/google.gemma-3-27b-it",
        metadata={
            "description": "Model used for generating responses. Format: 'provider/model-id'. Examples: 'bedrock_converse/us.anthropic.claude-3-5-sonnet-20241022-v2:0', 'bedrock_converse/us.amazon.nova-pro-v1:0', 'google_genai/gemini-2.5-flash'."
        },
    )

    # prompts

    generate_queries_system_prompt: str = field(
        default=prompts.GENERATE_QUERIES_SYSTEM_PROMPT,
        metadata={
            "description": "The system prompt used by the researcher to generate queries based on a step in the research plan."
        },
    )

    response_system_prompt: str = field(
        default=prompts.RESPONSE_SYSTEM_PROMPT,
        metadata={"description": "The system prompt used for generating responses."},
    )
