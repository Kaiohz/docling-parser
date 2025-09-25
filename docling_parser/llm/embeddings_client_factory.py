from langchain_google_genai import GoogleGenerativeAIEmbeddings
from llm.embeddings_generic_client import EmbeddingsGenericClient


class EmbeddingsClientFactory:
    def __init__(self, provider: str):
        self.provider = provider
        self.model_mapping = {
            "Google": GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        }

    def create_client(self) -> EmbeddingsGenericClient:
        client = self.model_mapping.get(self.provider)
        if not client:
            raise ValueError(f"Invalid provider name: {self.provider}")
        return client # type: ignore
