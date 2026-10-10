from typing import Any, Literal
from pydantic import BaseModel, Field, field_validator


DocType = Literal["invoice", "contract"]


class DocumentExtraction(BaseModel):
    filename: str = Field(..., description="Fișierul original")
    content: str = Field(..., min_length=1, description="Text complet extras")
    doc_type: DocType = Field(default="other")
    extra_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content cannot be empty or whitespace-only")
        return v

    @field_validator("doc_type", mode="before")
    @classmethod
    def validate_doc_type(cls, v: Any) -> str:
        # valori necunoscute → 'other' (în loc să crape validarea)
        allowed = {"invoice", "contract"}
        return v if v in allowed else "other"

    def to_db_format(self) -> dict[str, Any]:
        # convertește la formatul cerut de DocumentRepository.create()
        return {
            "filename": self.filename,
            "content": self.content,
            "metadata": {
            "doc_type": self.doc_type,
            **self.extra_metadata,
            },
        }


class ExtractionResult(BaseModel):
    success: bool
    document: DocumentExtraction | None = None
    error: str | None = None

class DocumentChunkExtraction(BaseModel):
    doc_type: DocType = "other"
    extra_metadata: dict[str, Any] = Field(default_factory=dict)