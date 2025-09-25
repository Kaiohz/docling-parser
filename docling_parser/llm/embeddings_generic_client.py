from langchain_core.embeddings import Embeddings
from pydantic import BaseModel


class EmbeddingsGenericClient(BaseModel, Embeddings):
    def __init__(self, model: str):
        super().__init__()
        self.model = model
