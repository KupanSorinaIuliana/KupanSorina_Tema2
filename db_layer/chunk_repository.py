from .models import DocumentChunk
from sqlalchemy import select, text
from sqlalchemy.orm import joinedload, Session

class ChunkRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    # WRITE
    def create_chunk(self, document_id: int, content: str, chunk_index: int, embedding: list[float]) -> DocumentChunk:
        chunk = DocumentChunk(
            document_id=document_id,
            content=content,
            chunk_index=chunk_index,
            embedding=embedding
        )
        self.db.add(chunk)
        self.db.flush()
        return chunk

    def create_chunks_batch(self, chunks: list[dict]) -> list[DocumentChunk]:
        chunk_objects = [DocumentChunk(**d) for d in chunks]
        self.db.add_all(chunk_objects)
        self.db.flush()
        return chunk_objects

    # READ
    def get_document_chunks(self, document_id: int) -> list[DocumentChunk]:
        return (
            self.db.query(DocumentChunk)
            .filter(DocumentChunk.document_id == document_id)
            .order_by(DocumentChunk.chunk_index)
            .all()
        )
    
    def get_chunk_by_id(self, chunk_id: int) -> DocumentChunk | None:
        return self.db.get(DocumentChunk, chunk_id)

    # SEARCH
    def similarity_search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        probes: int = 10,
    ) -> list[tuple[DocumentChunk, float]]:
        # 1 - (emb <=> q)  →  cosine distance convertit în similaritate [0, 1]
        similarity = (1 - DocumentChunk.embedding.cosine_distance(query_embedding)).label("score")

        self.db.execute(
            text("SELECT set_config('ivfflat.probes', :probes, true)"),
            {"probes": str(probes)},
        )

        stmt = (
            select(DocumentChunk, similarity)
            .options(joinedload(DocumentChunk.document))   # eager load — evită N+1
            .order_by(DocumentChunk.embedding.cosine_distance(query_embedding))
            .limit(top_k)
        )

        rows = self.db.execute(stmt).all()
        return [(chunk, float(score)) for chunk, score in rows]

    def search_with_threshold(self, query_embedding: list[float], min_score: float = 0.1, top_k: int = 5, probes: int = 10,) -> list[tuple[DocumentChunk, float]]:
        results = self.similarity_search(query_embedding, top_k=top_k, probes=probes)
        return [(chunk, score) for chunk, score in results if score >= min_score]
