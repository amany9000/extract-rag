"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated

from retrieval_graph import prompts
from shared.configuration import BaseConfiguration
from shared.utils import default_chat_model_name


@dataclass(kw_only=True)
class AgentConfiguration(BaseConfiguration):
    """The configuration for the agent."""

    # models

    query_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=default_chat_model_name,
        metadata={
            "description": "The language model used for processing and refining queries. Should be in the form: provider/model-name. Defaults to the model for the LLM_PROVIDER env var (gemini or bedrock)."
        },
    )

    response_model: Annotated[str, {"__template_metadata__": {"kind": "llm"}}] = field(
        default_factory=default_chat_model_name,
        metadata={
            "description": "The language model used for generating responses. Should be in the form: provider/model-name. Defaults to the model for the LLM_PROVIDER env var (gemini or bedrock)."
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
