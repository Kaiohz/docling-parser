from abc import ABC, abstractmethod
from typing import Any, Union
from dto.agent.agent_response import AgentResponse



class ResponseStrategy(ABC):
    """
    Interface for response processing strategies.
    """

    @abstractmethod
    async def execute(self, stream: Any) -> Union[AgentResponse, str]:
        """
        Processes the given stream and yields formatted responses.

        Args:
            stream (Any): The input stream to process.

        Yields:
            str: The formatted response.
        """
        pass
