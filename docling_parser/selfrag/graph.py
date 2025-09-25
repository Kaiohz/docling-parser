from langgraph.graph import END, StateGraph, START
from langgraph.graph.state import CompiledStateGraph
from dto.agent.agent import AgentParams
from selfrag.state import SelfRagState
from selfrag.nodes.nodes import SelfRAGNodes
from agent_interface import AlfredAgent


class AgentSelfRag(AlfredAgent):
    def __init__(self, agent_params: AgentParams):
        self.AlfredSelfRagNodes = SelfRAGNodes(agent_params=agent_params)

    def build_graph(self):
        builder = StateGraph(SelfRagState)

        builder.add_node("collection", self.AlfredSelfRagNodes.choose_collection)
        builder.add_node("retrieve", self.AlfredSelfRagNodes.retrieve)
        builder.add_node("grade_documents", self.AlfredSelfRagNodes.grade_documents)

        # Définir un point d'entrée conditionnel
        builder.set_conditional_entry_point(
            path=lambda state: "retrieve" if state.get("collections") else "collection",
            path_map={"retrieve": "retrieve", "collection": "collection"},
        )

        # Ajouter les transitions restantes
        builder.add_edge("collection", "retrieve")
        builder.add_edge("retrieve", "grade_documents")
        builder.add_edge("grade_documents", END)

        return builder

    def get_agent(self):
        """
        Compile and return the agent's state graph.

        Returns:
            StateGraph: The compiled agent graph.
        """
        app = self.build_graph().compile()
        app.name = "AgentSelfRag"
        return app
