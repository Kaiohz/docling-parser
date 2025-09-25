from dto.agent.agent_response import AgentResponse
from dto.rag.self_rag_response import SelfRagResponse
from response_strategy import ResponseStrategy
from selfrag.models.alfred_document import AlfredDocument
from shared.extract_step import extract_step_from_chunk


class SelfRagResponseStrategy(ResponseStrategy):
    """
    Strategy for handling responses from the SelfRag agent.
    This strategy processes the response and maps it to a SelfRagResponse object.
    """

    async def execute(self, chunk) -> AgentResponse: # type: ignore
        """
        Processes the SelfRag agent response and maps it to a SelfRagResponse object.

        Args:
            chunk (dict): The chunk of data containing SelfRag agent results.

        Returns:
            AgentResponse: The processed SelfRag response as an AgentResponse object.
        """
        response = AgentResponse() # type: ignore
        if chunk[1] and isinstance(chunk[1], dict):
            step = extract_step_from_chunk(chunk)
            if step:
                response.step = step
            if (
                "grade_documents" in chunk[1]
                and "documents" in chunk[1]["grade_documents"]
            ):
                documents: list[AlfredDocument] = chunk[1]["grade_documents"][
                    "documents"
                ]
                response.response = SelfRagResponse(sources=documents)
        return response
