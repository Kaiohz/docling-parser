import inspect
from langchain_core.vectorstores import VectorStoreRetriever

from dto.rag.retriever import RetrieverParams
from retrievers.self_query_retriever import CustomSelfQueryRetriever
from retrievers.multi_query_retriever import CustomMultiQueryRetriever
from retrievers.merge_retriever import CustomMergerRetriever


class RetrieverFactory:
    """
    Factory class for creating retrievers based on the given retriever name.
    """

    class_mapping = {
        "MultiQueryRetriever": CustomMultiQueryRetriever,
        "SelfQueryRetriever": CustomSelfQueryRetriever,
        "MergerRetriever": CustomMergerRetriever,
    }

    def __init__(self, retriever_params: RetrieverParams) -> None:
        self.retriever_name = retriever_params.retriever_name
        self.retriever_params = retriever_params

    def create_retriever(
        self, vectorstore, llm, filter_folder=None
    ) -> VectorStoreRetriever:
        """
        Creates a retriever based on the given retriever name.

        Args:
            vectorstore: The vector store to be used by the retriever.
            llm: The language model to be used by the retriever (optional).

        Returns:
            A VectorStoreRetriever instance.

        Raises:
            ValueError: If the retriever name is invalid.
        """
        if self.retriever_name in self.class_mapping:
            retriever_class = self.class_mapping[self.retriever_name]
            # Use introspection to get the parameters of the __init__ method
            parameters = inspect.signature(retriever_class.__init__).parameters
            # Check if 'llm' is one of the parameters
            if "llm" in parameters:
                return retriever_class(
                    vectorstore, llm, self.retriever_params, filter_folder
                ).get_retriever()
            else:
                return retriever_class(
                    vectorstore, self.retriever_params, filter_folder
                ).get_retriever()
        else:
            raise ValueError("Invalid retriever name")
