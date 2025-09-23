from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from typing import Type
from docling_parser.llm_generic_client import LLMGenericClient


class LLMClientFactory:
    def __init__(self, model: str):
        self.model = model
        self.temperature = 0.0
        self.ollama_base_url = "http://localhost:11434"
        self.open_router_base_url = "http://localhost:8000"
        self.open_router_api_key = "your_open_router_api_key"
        self.model_mapping: dict[str, Type[LLMGenericClient]] = {  # type: ignore
            "Ollama": ChatOllama,
            "OpenRouter": ChatOpenAI
        }

    def create_client(self) -> LLMGenericClient:
        category = self.model.split("/")[0]
        model = self.model.split("/", 1)[1]
        params = {"model": model, "temperature": self.temperature}
        client = self.model_mapping.get(category)
        if not client:
            raise ValueError(f"Invalid model name: {self.model}")
        if category == "Ollama":
            params["base_url"] = self.ollama_base_url
        if category == "OpenRouter":
            params["api_key"] = self.open_router_api_key
            params["base_url"] = self.open_router_base_url
        return client(**params)
