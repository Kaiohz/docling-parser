from langchain_core.vectorstores import VectorStore
from langchain.chains.query_constructor.base import AttributeInfo # type: ignore
from langchain.retrievers.self_query.base import SelfQueryRetriever

from dto.rag.retriever import RetrieverParams


class CustomSelfQueryRetriever:
    """
    A self-querying retriever generates and applies its own structured queries from natural language
    inputs using an LLM chain. It then uses these queries on its VectorStore to perform
    semantic similarity comparisons and metadata filtering for more precise search results.
    """

    def __init__(
        self,
        vectore_store: VectorStore,
        llm: str,
        retriever_params: RetrieverParams,
        filter_folder: str = None, # type: ignore
        metadata_field_info: list[AttributeInfo] = [],
    ) -> None:
        self.vector_store = vectore_store
        self.llm = llm
        self.filter = filter_folder
        self.metadata_field_info = metadata_field_info
        self.document_content_description = "Confluence page content"
        self.top_k = retriever_params.top_k
        self.score_threshold = retriever_params.score_threshold

    def get_retriever(self) -> SelfQueryRetriever:
        search_kwargs = {"score_threshold": self.score_threshold, "k": self.top_k}

        if self.filter:
            search_kwargs["filter"] = {"file_folder": self.filter}

        return SelfQueryRetriever.from_llm(
            self.llm, # type: ignore
            self.vector_store,
            self.document_content_description,
            self.metadata_field_info,
            search_kwargs=search_kwargs,
        )
