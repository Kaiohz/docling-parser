from pydantic import BaseModel, Field
from typing import Optional


class Metadata(BaseModel):
    """Metadata about the source of the document."""

    url: str = Field(..., description="The URL of the source")
    title: str = Field(..., description="The title of the source")
    when: Optional[str] = Field(
        None,
        description="The date and time when the source was created or last modified",
    )
    folder: Optional[str] = Field(
        None, description="The folder or directory where the source is located"
    )


class AlfredDocument(BaseModel):
    """Represents a document with metadata and content."""

    metadata: Metadata = Field(..., description="Metadata about the document")
    content: str = Field(
        ..., description="The content of the document, such as text or HTML"
    )
