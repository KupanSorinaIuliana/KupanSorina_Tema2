from typing import Any
from sqlalchemy.orm import Session
from sqlalchemy import func, update as sql_update, delete as sql_delete
from db_layer.models import Document
from db_layer.exceptions import DocumentNotFoundError, DuplicateDocumentError


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    # CREATE
    def create(self, filename: str, content: str, metadata: dict[str, Any]) -> Document:
        doc = Document(filename=filename, content=content, doc_metadata=metadata)  # 1. creează obiect
        self.db.add(doc)        # 2. adaugă în session (nu în DB încă)
        self.db.flush()        # 3. scrie în DB
        return doc
    
    def create_batch(self, documents: list[dict[str, Any]]) -> list[Document]:
        docs = [Document(**d) for d in documents]
        self.db.add_all(docs)   # un singur INSERT batch, mult mai rapid decât N add()
        self.db.flush()
        return docs

    # READ
    def get_by_id(self, doc_id: int) -> Document | None:
        # API-ul modern SQLAlchemy: Session.get() folosește direct PK-ul
        return self.db.get(Document, doc_id)

    def get_by_filename(self, filename: str) -> Document | None:
        return self.db.query(Document).filter(Document.filename == filename).first()

    def get_all(self, skip: int = 0, limit: int = 100) -> list[Document]:
        return (
            self.db.query(Document)
            .order_by(Document.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def count(self) -> int:
        return self.db.query(func.count(Document.id)).scalar()

    def filter_by_metadata(self, key: str, value: str) -> list[Document]:
        # operatorul ->> din PostgreSQL: extrage field din JSONB ca text
        return (
            self.db.query(Document)
            .filter(Document.doc_metadata[key].astext == value)
            .all()
        )

    def search(self, query: str, limit: int = 20) -> list[Document]:
        # tsquery construit cu același dicționar ca search_vector → stemming consistent
        ts_query = func.plainto_tsquery("romanian", query)
        return (
            self.db.query(Document)
            .filter(Document.search_vector.op("@@")(ts_query))
            .order_by(func.ts_rank(Document.search_vector, ts_query).desc())
            .limit(limit)
            .all()
        )

    # UPDATE
    def update(self, doc_id: int, **fields: Any) -> Document | None:
        clean = {k: v for k, v in fields.items() if k in self._UPDATABLE}
        if not clean:
            return self.db.get(Document, doc_id)
        self.db.execute(sql_update(Document).where(Document.id == doc_id).values(**clean))
        self.db.commit()
        return self.db.get(Document, doc_id)

    def update_metadata(self, doc_id: int, metadata: dict[str, Any]) -> Document | None:
        # JSONB merge la nivel de DB (atomic) — operatorul || din PostgreSQL
        self.db.execute(
            sql_update(Document)
            .where(Document.id == doc_id)
            .values(doc_metadata=Document.doc_metadata.op("||")(metadata))
        )
        self.db.commit()
        return self.db.get(Document, doc_id)

    # DELETE
    def delete(self, doc_id: int) -> bool:
        result = self.db.execute(sql_delete(Document).where(Document.id == doc_id))
        self.db.commit()
        return result.rowcount > 0

    def delete_batch(self, doc_ids: list[int]) -> int:
        if not doc_ids:
            return 0
        result = self.db.execute(sql_delete(Document).where(Document.id.in_(doc_ids)))
        self.db.commit()
        return result.rowcount

    def get_by_id_safe(self, doc_id: int) -> Document:
        doc = self.db.get(Document, doc_id)
        if doc is None:
            raise DocumentNotFoundError(doc_id)
        return doc

    def create_safe(self, filename: str, content: str, metadata: dict) -> Document:
        if self.get_by_filename(filename) is not None:
            raise DuplicateDocumentError(filename)
        if not isinstance(metadata, dict):
            raise InvalidMetadataError("metadata must be a dict")
        return self.create(filename=filename, content=content, metadata=metadata)
