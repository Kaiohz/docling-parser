from pydantic import BaseModel, Field
from typing import List

from agentapi.agents.selfrag.models.alfred_document import AlfredDocument


class SelfRagResponse(BaseModel):
    """Represents the response for a web search."""

    sources: List[AlfredDocument] = Field(
        ..., description="List of sources with details"
    )
