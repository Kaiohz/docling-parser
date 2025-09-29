from langchain_postgres import PGVector
import pymupdf
from llm.embeddings_client_factory import EmbeddingsClientFactory
from parser import get_toc_map, transform_document
import asyncio
import os

os.environ["GOOGLE_API_KEY"] = "key"

source = [
    "docs/Connect-RH-ADP.pdf",
    "docs/Connect-RH-PAIE.pdf",
    "docs/Connect-RH-REC.pdf",
    "docs/Connect-RH-RHPM.pdf"
]

async def main():

    embeddings = EmbeddingsClientFactory(provider="Google", model="models/text-embedding-004").create_client()

    connection_string = "postgresql+psycopg://docling:docling@localhost:5432/docling"

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="connect_rh_adp",
        connection=connection_string,
        use_jsonb=True,
        async_mode=True,
        pre_delete_collection=True
    )

    for doc in source:
        doc = pymupdf.open(doc)
        page_title_map = await get_toc_map(doc)
        documents = await transform_document(doc, page_title_map)
        await vector_store.aadd_documents(documents)

if __name__ == "__main__":
    asyncio.run(main())