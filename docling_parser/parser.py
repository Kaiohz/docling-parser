from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAI
import asyncio

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
    model = GoogleGenerativeAI(model="gemini-2.0-flash")  # Adjust model name as needed

    async def process_page(page_number, page):
        title = page_title_map.get(page_number, "No Title")
        original_text = page.get_text()

        prompt = (
            "Remove any irrelevant context and keep only the main content from the following page.\n"
            "If the content is only section headers, navigation, or similar irrelevant text, return an empty string.\n"
            "Examples of irrelevant content:\n"
            "Recrutement\nRECRUTEMENT\nEMBAUCHE\nEMBAUCHE\nVie du CONTRAT\nVIE DU CONTRAT\n"
            "Sortie\nSORTIE\nEXTRACTION D’UN RAPPORT\nRetour\nRetour\n\n"
            "Another example of irrelevant content, no information valuable here:\n"
            "LE GRAND MANUEL\n"
            "Porte ouverte avec un remplissage uni\n"
            "Page content:\n"
            f"{original_text}\n\n"
            "Return only the relevant content. DO NOT ALTER THE CONTENT. IF THE CONTENT IS IRRELEVANT, RETURN AN EMPTY STRING."
        )

        response = await model.ainvoke(prompt)
        filtered_text = response.content if hasattr(response, "content") else str(response) # type: ignore

        prompt_subtitle = (
            "Extract the title of the page if it exists. If no title is found, return an empty string.\n"
            f"Page content:\n{filtered_text}\n\n"
        )

        response_subtitle = await model.ainvoke(prompt_subtitle)
        extracted_title = response_subtitle.content if hasattr(response_subtitle, "content") else str(response_subtitle) # type: ignore

        if filtered_text.strip():
            return Document(
                page_content=filtered_text,
                metadata={"category": title, "sub_category": extracted_title.strip(), "page_number": page_number}
            )
        return None

    tasks = [
        process_page(page_number, page)
        for page_number, page in enumerate(doc, start=1)  # type: ignore
    ]
    results = await asyncio.gather(*tasks)
    documents = [doc for doc in results if doc is not None]
    return documents