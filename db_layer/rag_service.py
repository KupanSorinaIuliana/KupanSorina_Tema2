from sentence_transformers import SentenceTransformer
from sqlalchemy.orm import Session
from .chunk_repository import ChunkRepository
from .models import DocumentChunk

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_THRESHOLD = 0.4


class RAGService:
    _model: SentenceTransformer | None = None   # singleton per proces

    def __init__(self, db: Session) -> None:
        self.repo = ChunkRepository(db)

    @property
    def model(self) -> SentenceTransformer:
        # lazy + shared: modelul (~80MB) se încarcă o singură dată per proces
        if RAGService._model is None:
            RAGService._model = SentenceTransformer(MODEL_NAME)
        return RAGService._model

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        query_emb = self.model.encode(query, convert_to_numpy=True).tolist()
        return self.repo.similarity_search(query_emb, top_k=top_k)

    def get_context(
        self,
        query: str,
        top_k: int = 5,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> str:
        results = self.search(query, top_k=top_k)
        relevant = [(chunk, score) for chunk, score in results if score >= threshold]
        if not relevant:
            return ""
        return "\n\n".join(
            f"[{chunk.document.filename}] {chunk.content}" for chunk, _ in relevant
        )
