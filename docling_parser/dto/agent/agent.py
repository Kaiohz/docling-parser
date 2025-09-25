from pydantic import BaseModel, Field
from typing import Any, List


class AgentParams(BaseModel):
    agent_name: str = Field(..., description="Agent name to run")
    input_data: Any = Field(..., description="Input data to run")
    model_name: str = Field(..., description="Model name to run")
    stream_mode: List[str] = Field(..., description="Stream mode to run")
