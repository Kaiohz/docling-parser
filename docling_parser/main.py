import re
from docling.document_converter import DocumentConverter
import pymupdf
import pymupdf4llm


source = "docs/24072025 Grand Manuel - Connect RH ADP.pdf"  # file path or URL
# converter = DocumentConverter()
# doc = converter.convert(source)

doc = pymupdf.open(source)
md_text = pymupdf4llm.to_markdown(source)
TOC = doc.get_toc()  # use the table of contents for determining headers

# Create a map with top title and table of pages (excluding top title page)

# Collect all top-level titles and their pages


# Build a string: "top title : page number" for each top-level title




# Build a list of strings: ["title : pageX", ...] for every page in each top-level section

# Build a map: {page_number: title} for every page in each top-level section
# page_title_map = {}
# if toc:
#     top_titles = [(title, page) for level, title, page in toc if level == 1]
#     section_ranges = []
#     for i, (title, start_page) in enumerate(top_titles):
#         if i + 1 < len(top_titles):
#             end_page = top_titles[i + 1][1] - 1
#         else:
#             end_page = None  # Last section goes to end of document
#         section_ranges.append((title, start_page, end_page))

#     all_pages = [page for _, _, page in toc]
#     max_page = max(all_pages) if all_pages else 0

#     for title, start_page, end_page in section_ranges:
#         if end_page is None:
#             pages = list(range(start_page, max_page + 1))
#         else:
#             pages = list(range(start_page, end_page + 1))
#         for p in pages:
#             page_title_map[p] = title
# else:
#     page_title_map = {}

# page_number = 0
# for page in doc:
#     page_number += 1
#     title = page_title_map.get(page_number, "No Title")
#     print(f"{title} : {page_number} : {page.get_text()[:20]}...")
#     print("-" * 40)

def my_headers(span, page=None):
    """
    Provide some custom header logic (experimental!).
    This callable checks whether the span text matches any of the
    TOC titles on this page.
    If so, use TOC hierarchy level as header level.
    """
    # TOC items on this page:
    toc = [t for t in TOC if t[-1] == page.number + 1]

    if not toc:  # no TOC items on this page
        return ""

    # look for a match in the TOC items
    for lvl, title, _ in toc:
        if span["text"].startswith(title):
            return "#" * lvl + " "
        if title.startswith(span["text"]):
            return "#" * lvl + " "

    return ""

# this will *NOT* scan the document for font sizes!
md_text = pymupdf4llm.to_markdown(doc, hdr_info=my_headers)


print(md_text[:2000])  # print first 2000 characters of markdown text

doc.close()


# toc_map now contains the desired structure