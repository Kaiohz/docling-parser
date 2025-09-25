from langchain_core.vectorstores import VectorStore
from langchain.retrievers import (
    MergerRetriever,
)
from dto.rag.retriever import RetrieverParams
from retrievers.multi_query_retriever import CustomMultiQueryRetriever
from retrievers.self_query_retriever import CustomSelfQueryRetriever


class CustomMergerRetriever:
    """
    Lord of the Retrievers (LOTR), also known as MergerRetriever, takes a list of retrievers
    as input and merges the results of their get_relevant_documents() methods into a single list.
    The merged results will be a list of documents that are relevant to the query and that have
    been ranked by the different retrievers.
    """

    def __init__(
        self,
        vectore_store: VectorStore,
        llm: str,
        retriever_params: RetrieverParams,
        filter_folder: str = None, # type: ignore
    ) -> None:
        self.vector_store = vectore_store
        self.llm = llm
        self.filter = filter_folder
        self.self_query_retriever = CustomSelfQueryRetriever(
            llm=self.llm,
            vectore_store=self.vector_store,
            retriever_params=retriever_params,
            filter_folder=self.filter,
        ).get_retriever()
        self.vector_store_retriever = CustomMultiQueryRetriever(
            llm=self.llm,
            vectore_store=self.vector_store,
            retriever_params=retriever_params,
            filter_folder=self.filter,
        ).get_retriever()

    def get_retriever(self) -> MergerRetriever:
        return MergerRetriever(
            retrievers=[self.self_query_retriever, self.vector_store_retriever]
        )
