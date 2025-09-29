from pydantic import BaseModel, Field


class GradeDocuments(BaseModel):
    """Score for relevance check on retrieved documents."""

    relevance_score: int = Field(
        description="Relevance score of the retrieved document based on the question. from 0 to 100"
    )
