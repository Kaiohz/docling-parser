from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_ollama import OllamaEmbeddings
from llm.embeddings_generic_client import EmbeddingsGenericClient


class EmbeddingsClientFactory:
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model_mapping = {
            "Google": GoogleGenerativeAIEmbeddings(model=model),
            "Ollama": OllamaEmbeddings(model=model),
        }

    def create_client(self) -> EmbeddingsGenericClient:
        client = self.model_mapping.get(self.provider)
        if not client:
            raise ValueError(f"Invalid provider name: {self.provider}")
        return client # type: ignore
