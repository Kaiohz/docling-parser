from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from llm.chat_generic_client import ChatGenericClient
from langchain_ollama import ChatOllama


class ChatClientFactory:
    def __init__(self, provider: str, temperature: float, model: str):
        self.provider = provider
        self.temperature = temperature
        self.model_mapping = {
            "Google": ChatGoogleGenerativeAI(model=model),
            "Ollama": ChatOllama(model=model),

        }

    def create_client(self) -> ChatGenericClient:
        client = self.model_mapping.get(self.provider)
        if not client:
            raise ValueError(f"Invalid model name: {self.provider}")
        return client # type: ignore
