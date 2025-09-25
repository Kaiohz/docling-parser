from langchain_google_genai import ChatGoogleGenerativeAI
from llm.chat_generic_client import ChatGenericClient


class ChatClientFactory:
    def __init__(self, provider: str, temperature: float):
        self.provider = provider
        self.temperature = temperature
        self.model_mapping = {
            "Google": ChatGoogleGenerativeAI(model="gemini-2.0-flash")
        }

    def create_client(self) -> ChatGenericClient:
        client = self.model_mapping.get(self.provider)
        if not client:
            raise ValueError(f"Invalid model name: {self.provider}")
        return client # type: ignore
