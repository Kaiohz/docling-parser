### Retrieval Grader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from langchain.embeddings import CacheBackedEmbeddings
from llm.chat_client_factory import ChatClientFactory
from selfrag.models.grade_document import GradeDocuments
from selfrag.models.self_rag_input import SelfRagInput
from retrievers.rag_retriever_factory import RetrieverFactory
from langchain.storage import InMemoryByteStore
from util.prompt_loader import prompts


store = InMemoryByteStore()


class RetrievalChain:
    def __init__(self, model: str, ask_params: SelfRagInput):
        self.llm = ChatClientFactory(
            provider="Google", temperature=0.0
        ).create_client()
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts["retrieval_grader"]),
                (
                    "human",
                    "Retrieved document: \n\n {document} \n\n User question: {question}",
                ),
            ]
        )
        self.retrieval_grader = self.grade_prompt | self.structured_llm_grader
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        self.cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            self.embeddings, store, namespace=ask_params.embeddings_model
        )
        connection_string = "postgresql+psycopg://docling:docling@localhost:5432/docling"

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="connect_rh_adp",
            connection=connection_string,
            use_jsonb=True,
            async_mode=True,
        )
        
        self.retriever = RetrieverFactory(ask_params).create_retriever( # type: ignore
            self.vector_store, self.llm
        )
