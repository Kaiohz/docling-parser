from langchain_core.vectorstores import VectorStore
from langchain.retrievers.multi_query import MultiQueryRetriever
from dto.rag.retriever import RetrieverParams


class CustomMultiQueryRetriever:
    """
    The MultiQueryRetriever automates the process of prompt tuning by using an LLM to generate
    multiple queries from different perspectives for a given user input query.
    For each query, it retrieves a set of relevant documents and takes the unique union across
    all queries to get a larger set of potentially relevant documents.
    By generating multiple perspectives on the same question, the MultiQueryRetriever might be able
    to overcome some of the limitations of the distance-based retrieval and get a richer set of results.
    """

    def __init__(
        self,
        vectore_store: VectorStore,
        llm: str,
        retriever_params: RetrieverParams,
        filter_folder: str = None, # type: ignore
    ) -> None:
        self.llm = llm
        self.score_threshold = retriever_params.score_threshold
        self.top_k = retriever_params.top_k
        self.vector_store = vectore_store
        self.filter = filter_folder

    def get_retriever(self) -> MultiQueryRetriever:
        search_kwargs = {"score_threshold": self.score_threshold, "k": self.top_k}

        if self.filter:
            search_kwargs["filter"] = {"file_folder": self.filter}

        return MultiQueryRetriever.from_llm(
            retriever=self.vector_store.as_retriever(
                search_type="similarity_score_threshold", search_kwargs=search_kwargs
            ),
            llm=self.llm, # type: ignore
        )
