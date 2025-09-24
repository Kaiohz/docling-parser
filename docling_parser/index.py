from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
import pymupdf
from parser import get_toc_map, transform_document
from retriever import ask_vector_store
import asyncio
import os

source = "docs/sample.pdf" 
os.environ["GOOGLE_API_KEY"] = "key"


async def main():

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    connection_string = "postgresql+psycopg://docling:docling@localhost:5432/docling"

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name="connect_rh_adp",
        connection=connection_string,
        use_jsonb=True,
        async_mode=True,
        pre_delete_collection=True,
    )

    doc = pymupdf.open(source)
    page_title_map = await get_toc_map(doc)
    documents = await transform_document(doc, page_title_map)
    await vector_store.aadd_documents(documents)
    # toc_map now contains the desired structure

if __name__ == "__main__":
    asyncio.run(main())