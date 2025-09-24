from multiprocessing import connection
import pymupdf
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres.vectorstores import PGVector
import os
import asyncio

os.environ["GOOGLE_API_KEY"] = ""
source = "" 

doc = pymupdf.open(source)
toc = doc.get_toc() # type: ignore

page_title_map = {}
if toc:
    top_titles = [(title, page) for level, title, page in toc if level == 1]
    section_ranges = []
    for i, (title, start_page) in enumerate(top_titles):
        if i + 1 < len(top_titles):
            end_page = top_titles[i + 1][1] - 1
        else:
            end_page = None  # Last section goes to end of document
        section_ranges.append((title, start_page, end_page))

    all_pages = [page for _, _, page in toc]
    max_page = max(all_pages) if all_pages else 0

    for title, start_page, end_page in section_ranges:
        if end_page is None:
            pages = list(range(start_page, max_page + 1))
        else:
            pages = list(range(start_page, end_page + 1))
        for p in pages:
            page_title_map[p] = title
else:
    page_title_map = {}

documents = []
for page_number, page in enumerate(doc, start=1): # type: ignore
    title = page_title_map.get(page_number, "No Title")
    documents.append(
        Document(
            page_content=page.get_text(),
            metadata={"title": title}
        )
    )

embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
embedded_docs = embeddings.embed_documents([doc.page_content for doc in documents])

connection_string = "postgresql+psycopg://docling:docling@localhost:5432/docling"

vector_store = PGVector(
    embeddings=embeddings,
    collection_name="connect_rh_adp",
    connection=connection_string,
    use_jsonb=True,
    pre_delete_collection=True,
)

vector_store.add_documents(documents)

print("Documents added to the vector store.")
# toc_map now contains the desired structure