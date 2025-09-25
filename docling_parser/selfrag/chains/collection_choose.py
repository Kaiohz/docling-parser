### Generate
from llm.chat_client_factory import ChatClientFactory
from selfrag.models.collections import MatchingCollections
from util.prompt_loader import prompts
from langchain_core.prompts import ChatPromptTemplate


class CollectionChooseChain:

    def __init__(self, model: str, graph_name: str):
        self.llm = ChatClientFactory(
            provider="Google", temperature=0.0
        ).create_client()

        self.structured_llm_collection = self.llm.with_structured_output(
            MatchingCollections
        )
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts[f"{graph_name}_collection_choose"]),
                ("human", "{question}"),
            ]
        )
        self.collection_choose = self.prompt | self.structured_llm_collection
