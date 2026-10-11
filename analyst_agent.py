import os

from pydantic import BaseModel
from google import genai
from db_layer.rag_service import RAGService
from db_layer.models import DocumentChunk
from db_layer.transaction_manager import transaction
import logging
from dotenv import load_dotenv

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL")
PROMPT_TEMPLATE = """Răspunde la întrebare BAZAT EXCLUSIV pe contextul de mai jos.

Reguli:
- Citează clauza/documentul folosit (ex: „conform contract.pdf, Clauza 5.3")
- Dacă răspunsul nu e în context, spune explicit: „Nu am informații despre asta în documente"
- Răspuns concis, fără padding

Context:
{context}

Întrebare: {question}
"""

class AnswerResult(BaseModel):
    answer: str
    sources: list[dict]      # [{filename, chunk_index, score}, ...]
    confidence: float        # 0.0 - 1.0


class AnalystAgent:
    def __init__(self, rag: RAGService, model: str = "gemini-3.8-flash") -> None:
        self.rag = rag
        self.model = model
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def answer_question(self, question: str, top_k: int = 5) -> AnswerResult:
        results = self.rag.search(question, top_k=top_k)
        if not results:
            return AnswerResult(
                answer="Nu am informații despre asta în documente",
                sources=[],
                confidence=0.0,
            )

        context = self._build_context(results)
        prompt = self._build_prompt(question, context)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        answer = getattr(response, "text", "").strip()
        if not answer:
            answer = "Nu am informații despre asta în documente"

        return AnswerResult(
            answer=answer,
            sources=self._extract_sources(results),
            confidence=self._calculate_confidence(results),
        )

    def _build_context(
        self,
        results: list[tuple[DocumentChunk, float]],
    ) -> str:
        # chunks → context string, cu score-uri inline pentru transparență
        parts = []
        for chunk, score in results:
            parts.append(
                f"[{chunk.document.filename} | chunk {chunk.chunk_index} | score {score:.2f}]\n"
                f"{chunk.content}"
            )
        return "\n\n".join(parts)

    def _build_prompt(self, question: str, context: str) -> str:
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def _extract_sources(self, results: list[tuple[DocumentChunk, float]]) -> list[dict]:
        # chunks → metadata pentru citare
        return [
            {
                "filename": chunk.document.filename,
                "chunk_index": chunk.chunk_index,
                "score": score,
            }
            for chunk, score in results
        ]

    def _calculate_confidence(self, results: list[tuple[DocumentChunk, float]]) -> float:
        # avg similarity score → 0.0 - 1.0
        if not results:
            return 0.0
        average = sum(score for _, score in results) / len(results)
        return max(0.0, min(1.0, average))

def main():
    # Show your application's warnings and errors, but not routine SQL.
    logging.basicConfig(level=logging.WARNING)

    # Keep your own application logs at INFO if desired.
    logging.getLogger("agents").setLevel(logging.INFO)
    
    # Reduce third-party library logging.
    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)
    
    # Disable Transformers progress bars.
    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()

    # Usage
    with transaction() as db:
        rag = RAGService(db)
        agent = AnalystAgent(rag, model=MODEL)

        for q in [
            "Ce servicii are factura lui Boldor Razvan?",
        ]:
            print(f"Q: {q}")
            print(f"A: {agent.answer_question(q)}\n")


if __name__ == "__main__":
    main()