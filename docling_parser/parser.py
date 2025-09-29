from langchain.schema import Document
from langchain_core.messages import HumanMessage
import base64
import pymupdf4llm
import pymupdf
import asyncio
import os
import uuid 

from docling_parser.llm.chat_client_factory import ChatClientFactory

async def get_toc_map(doc) -> dict:
    toc = doc.get_toc()  # type: ignore

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
    
    return page_title_map

async def transform_document(doc, page_title_map) -> list:
    documents = []
    semaphore = asyncio.Semaphore(100)
    model = ChatClientFactory(provider="Google", temperature=0.0, model="gemini-2.0-flash").create_client()

    async def process_page(page_number, page):
        async with semaphore:
            title = page_title_map.get(page_number, "No Title")
            temp_doc = pymupdf.open()
            temp_doc.insert_pdf(doc, from_page=page_number, to_page=page_number)
            pix = page.get_pixmap()
            temp_pdf = f"temp_page_{uuid.uuid4()}.pdf"
            temp_image = f"temp_image_{uuid.uuid4()}.png"
            pix.save(temp_image)
            temp_doc.save(temp_pdf)
            temp_doc.close()

            with open(temp_pdf, "rb") as f:
                pdf_base64 = base64.b64encode(f.read()).decode()

            with open(temp_image, "rb") as f:
                image_base64 = base64.b64encode(f.read()).decode()
            
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "Describe the following image in detail. Answer in Markdown format."},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                    }
                ]
            )

            response = await model.ainvoke([message]) # type: ignore

            md_text = pymupdf4llm.to_markdown(temp_pdf)
            os.remove(temp_pdf)
            os.remove(temp_image)

            print(f"Converted {title} page {page_number} to markdown")

            if md_text.strip():
                return Document(
                    page_content=md_text + "\n\n" + response.content, # type: ignore
                    metadata={"category": title, "page_number": page_number, "pdf_base64": f"{pdf_base64}"} # type: ignore
                )
            return None

    tasks = [
        process_page(page_number, page)
        for page_number, page in enumerate(doc, start=1)  # type: ignore
    ]
    results = await asyncio.gather(*tasks)
    documents = [doc for doc in results if doc is not None]
    return documents