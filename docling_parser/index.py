from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
import pymupdf
from parser import get_toc_map, transform_document
import asyncio
import os

source = [
    "docs/Connect-RH-ADP.pdf",
    "docs/Connect-RH-PAIE.pdf",
    "docs/Connect-RH-REC.pdf",
    "docs/Connect-RH-RHPM.pdf"
] 
os.environ["GOOGLE_API_KEY"] = ""


async def main():

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    connection_string = "postgresql+psycopg://docling:docling@localhost:5432/docling"

    for doc in source:
        vector_store = PGVector(
            embeddings=embeddings,
            collection_name=doc.split("/")[1].split(".")[0].lower(),
            connection=connection_string,
            use_jsonb=True,
            async_mode=True,
            pre_delete_collection=True,
        )

        doc = pymupdf.open(doc)
        page_title_map = await get_toc_map(doc)
        documents = await transform_document(doc, page_title_map)
        await vector_store.aadd_documents(documents)

if __name__ == "__main__":
    asyncio.run(main())