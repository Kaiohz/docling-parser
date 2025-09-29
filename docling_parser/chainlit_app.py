import chainlit as cl
from ask import main as asking_gent
import base64
import pymupdf
import uuid

@cl.on_message
async def main(message: cl.Message):
    answer, metadatas = await asking_gent(message.content)
    elements=[]
    merged_pdf = pymupdf.open()
    # sort metadatas by page_number
    metadatas = sorted(metadatas, key=lambda x: x.get('page_number', 0))
    # merge pdfs
    for metadata in metadatas:
        pdf_bytes = base64.b64decode(metadata['pdf_base64'])
        page = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        merged_pdf.insert_pdf(page)
        page.close()

    if merged_pdf.page_count > 0:
        merged_pdf_bytes = merged_pdf.write()
        merged_pdf.close()
        elements = [
            cl.Pdf(name="pdf", display="inline", content=merged_pdf_bytes, size="large")
        ]

    await cl.Message(
        content=answer,
        elements=elements,
        author="ConnectRH"
    ).send()