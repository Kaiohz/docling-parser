from langchain_core.language_models.chat_models import BaseChatModel


class ChatGenericClient(BaseChatModel):
    def __init__(self, model: str, temperature: float):
        super().__init__()
        self.model = model
        self.temperature = temperature
