from abc import ABC, abstractmethod
from langgraph.graph.state import CompiledStateGraph


class AlfredAgent(ABC):
    """
    Abstract base class for all agents.
    """

    @abstractmethod
    def get_agent(self) -> CompiledStateGraph:
        """
        Returns the compiled state graph for the agent.

        Returns:
            CompiledStateGraph: The compiled state graph instance for the agent.
        """
        pass
