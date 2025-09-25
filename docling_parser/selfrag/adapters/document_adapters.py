from langchain_core.documents import Document
from selfrag.models.alfred_document import AlfredDocument, Metadata


class DocumentAdapters:

    def __init__(self, graph_name: str):
        self.graph_name = graph_name
        self.mapping_document = {
            "AlfredSelfRag": self.alfred_selfrag_convert,
            "NotionSelfRag": self.alfred_selfrag_convert,
        }

    def adapt(self, documents: list[Document]) -> list[AlfredDocument]:
        return self.mapping_document[self.graph_name](documents)

    def alfred_selfrag_convert(self, documents: list[Document]) -> list[AlfredDocument]:
        alfred_documents = []
        for doc in documents:
            metadata = Metadata(
                url=doc.metadata.get("source"),
                title=doc.metadata.get("title"),
                when=doc.metadata.get("when"),
            ) # type: ignore
            alfred_document = AlfredDocument(
                metadata=metadata, content=doc.page_content
            )
            alfred_documents.append(alfred_document)
        return alfred_documents
