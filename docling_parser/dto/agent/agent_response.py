from pydantic import BaseModel, Field
from typing import Union
from agentapi.agents.dto.websearch.web_search_response import WebSearchResponse
from agentapi.agents.dto.maria.maria_response import MariaResponse


class AgentResponse(BaseModel):
    step: str = Field(None, description="step name of the agent")
    response: Union[WebSearchResponse, MariaResponse] = Field(
        None,
        description="Response from the agent, can be a web search response or other types",
    )
