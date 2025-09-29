from typing import List, Optional
from typing_extensions import TypedDict
from langchain_core.documents import Document
from selfrag.models.self_rag_input import SelfRagInput


class SelfRagState(TypedDict):
    """
    Represents the state for the SelfRag agent.

    Attributes:
        step (str): The current step or phase in the agent's process.
        documents (Optional[List[AlfredDocument]]): A list of AlfredDocument objects relevant to the current state, or None if not set.
        collections (Optional[List[str]]): A list of collection names associated with the current state, or None if not set.
    """

    step: str
    input_data: SelfRagInput  # Contains the search query
    documents: Optional[List[Document]]