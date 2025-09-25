from selfrag.graph import AgentSelfRag
from dto.agent.agent import AgentParams
from agent_interface import AlfredAgent


class AgentFactory:
    """
    Factory class for creating different types of agents.
    """

    class_mapping = {
        "AgentSelfRag": AgentSelfRag
    }

    def __init__(self, agent_name: str):
        """
        Initializes the AgentFactory with the specified agent name.

        Args:
            agent_name (str): The name of the agent to create.
        """
        self.agent_name = agent_name

    def create_agent(self, agent_params: AgentParams) -> AlfredAgent:
        """
        Creates an agent instance based on the specified agent name and parameters.

        Args:
            agent_params (AgentParams): The parameters to initialize the agent.

        Returns:
            AlfredAgent: The created agent instance.

        Raises:
            ValueError: If the agent name is invalid.
        """
        if self.agent_name in self.class_mapping:
            return self.class_mapping[self.agent_name](agent_params)
        else:
            raise ValueError("Invalid agent name")
