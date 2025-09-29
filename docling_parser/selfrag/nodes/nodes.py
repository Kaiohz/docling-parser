import asyncio
from typing import List
from dto.agent.agent import AgentParams
from selfrag.state import SelfRagState
from selfrag.chains.retrieval_grader import RetrievalChain
from langchain_core.documents import Document
from selfrag.models.self_rag_input import SelfRagInput
from util.logger import logger
import traceback


class SelfRAGNodes:

    _semaphore = asyncio.Semaphore(100)

    def __init__(self, agent_params: AgentParams):
        """
        Initialize the SelfRAGNodes with the required parameters and node dependencies.

        Args:
            ask_params (AskParams): Parameters for the current ask/query, including model and graph name.
        """
        self.ask_params = SelfRagInput(**agent_params.input_data)
        self.completion_model = agent_params.model_name

    async def retrieve(self, state: SelfRagState) -> SelfRagState:
        """
        Retrieve documents from the selected collections using the retrieval node.

        Args:
            state (dict): The current graph state containing 'question' and 'collections'.

        Returns:
            GraphState: Updated graph state with the retrieved documents.
        """
        try:
            question = self.ask_params.question
            retrieval_node = RetrievalChain(
                model=self.completion_model, ask_params=self.ask_params
            )
            documents = await retrieval_node.retriever.ainvoke(question)
            return SelfRagState(step="Filtrage des documents", documents=documents) # type: ignore
        except Exception as e:
            logger.error(f"===== SELFRAG AGENT: ERROR IN RETRIEVE =====")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return SelfRagState(step="Filtrage des documents", documents=[]) # type: ignore

    async def grade_documents(self, state: SelfRagState) -> SelfRagState:
        """
        Evaluate the relevance of each retrieved document to the user's question and filter accordingly.

        Args:
            state (dict): The current graph state containing 'question' and 'documents'.

        Returns:
            GraphState: Updated graph state with only relevant, adapted documents.
        """
        try:
            question = self.ask_params.question
            documents: List[Document] = state["documents"] # type: ignore
            filtered_docs: List[Document] = []

            async def get_score(d):
                async with self._semaphore:
                    retrieval_node = RetrievalChain(
                        ask_params=self.ask_params, model=self.completion_model
                    )
                    score = await retrieval_node.retrieval_grader.ainvoke(
                        {"question": question, "document": d}
                    )
                    return score.relevance_score, d # type: ignore

            tasks = [get_score(d) for d in documents]
            results = await asyncio.gather(*tasks)
            results.sort(key=lambda x: x[0], reverse=True)
            filtered_docs = [
                d for grade, d in results if grade >= 80
            ]
            return SelfRagState(
                step="Filtrage fini, preparation de la reponse", documents=filtered_docs
            ) # type: ignore
        except Exception as e:
            logger.error(f"===== SELFRAG AGENT: ERROR IN GRADE_DOCUMENTS =====")
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return SelfRagState(
                step="Filtrage fini, preparation de la reponse", documents=[]
            ) # type: ignore
