from pydantic import BaseModel, Field


class SelfRagInput(BaseModel):
    """Parameters for asking the vector store"""

    question: str = Field(..., description="Query question to ask the vector store")
    graph_name: str = Field("AlfredSelfRag", description="The name of the graph")
    embeddings_model: str = Field(
        "azure/text-embedding-ada-002", description="Embeddings model name"
    )
    retriever_name: str = Field(
        "MergerRetriever", description="The name of the retriever"
    )
    score_threshold: float = Field(
        0.8, description="The score threshold for the retriever"
    )
    top_k: int = Field(5, description="The number of documents to retrieve")
