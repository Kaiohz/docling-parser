from pydantic import BaseModel, Field
from typing import Optional


class RetrieverParams(BaseModel):
    """Parameters for asking the vector store"""

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
