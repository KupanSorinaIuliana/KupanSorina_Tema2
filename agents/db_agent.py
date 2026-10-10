import logging
import os
from pathlib import Path
from typing import Any

from google import genai
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from db_layer.document_extractor import (
    DocumentExtraction,
    DocumentChunkExtraction,
    ExtractionResult,
)
from db_layer.transaction_manager import transaction
from db_layer.document_repository import DocumentRepository
from db_layer.chunk_repository import ChunkRepository
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL")
EXTRACTION_PROMPT = """Extract structured metadata from the document below.

Filename: {filename}

Content:
{content}
"""

class DbAgent:
    def __init__(self, model: str = "gemini-3.8-flash",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_threshold: int = 4000,) -> None:
        self.model = model
        self.embedding_model_name = embedding_model

        self._client: genai.Client | None = None  # lazy
        self._embedding_model: SentenceTransformer | None = None

        self.chunk_threshold = chunk_threshold
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def client(self) -> genai.Client:
        # init per proces — clientul nu e pickle-able
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def split_content(self, content: str) -> list[str]:
        if not content.strip():
            return []

        if len(content) <= self.chunk_threshold:
            return [content]

        return self.splitter.split_text(content)

    def generate_embeddings(
        self,
        chunks: list[str],
    ) -> list[list[float]]:
        if not chunks:
            return []

        vectors = self.embeddings_model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if vectors.shape[1] != 384:
            raise ValueError(
                f"Expected 384-dimensional embeddings, "
                f"got {vectors.shape[1]}"
            )

        return vectors.tolist()

    def _generate_json_for_chunk(
            self,
            chunk_text: str,
            filename: str,
        ) -> dict[str, Any]:
            prompt = EXTRACTION_PROMPT.format(
                filename=filename,
                content=chunk_text,
            )
    
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": (
                        DocumentChunkExtraction.model_json_schema()
                    ),
                },
            )
    
            parsed = response.parsed
    
            if parsed is None:
                raise ValueError(
                    f"Empty Gemini response for {filename}"
                )
    
            if isinstance(parsed, BaseModel):
                parsed = parsed.model_dump()
    
            if not isinstance(parsed, dict):
                raise TypeError(
                    f"Unexpected Gemini response type: "
                    f"{type(parsed).__name__}"
                )
    
            # Explicit validation also handles SDK responses returned as dicts.
            validated = DocumentChunkExtraction.model_validate(parsed)
    
            return validated.model_dump()

    def _merge_json_objects(
        self,
        base: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(base)

        for key, value in incoming.items():
            if value is None:
                continue

            if key not in merged or merged[key] is None:
                merged[key] = value
                continue

            current = merged[key]

            # A later chunk may identify the document type more accurately.
            if key == "doc_type":
                if current == "other" and value != "other":
                    merged[key] = value
                continue

            if isinstance(current, dict) and isinstance(value, dict):
                merged[key] = self._merge_json_objects(
                    current,
                    value,
                )

            elif isinstance(current, list) and isinstance(value, list):
                combined = list(current)

                for item in value:
                    if item not in combined:
                        combined.append(item)

                merged[key] = combined

            # Conflicting scalar values preserve the first value.
            # Domain-specific reconciliation can be added for invoice fields.

        return merged
    
    @property
    def embeddings_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name
            )
        return self._embedding_model

    def load_document(self, path: str | Path) -> str | None:
        path = Path(path)
        try:
            match path.suffix.lower():
                case ".pdf":
                    loader = PyPDFLoader(str(path))
                case ".docx":
                    loader = Docx2txtLoader(str(path))
                case ".txt":
                    loader = TextLoader(str(path), autodetect_encoding=True)
                case _:
                    logger.warning("Unsupported format: %s", path.suffix)
                    return None

            docs = loader.load()
            return "\n\n".join(d.page_content for d in docs)  # combine pagini
        except Exception as e:
            logger.error("Failed to load %s: %s", path, e)
            return None

    def extract_metadata(
        self,
        content: str,
        filename: str,
    ) -> DocumentExtraction | None:
        try:
            chunks = self.split_content(content)

            if not chunks:
                logger.error(
                    "No chunks generated for %s",
                    filename,
                )
                return None

            logger.info(
                "Extracting %s: %d chunks, %d characters",
                filename,
                len(chunks),
                len(content),
            )

            merged_payload: dict[str, Any] = {
                "doc_type": "other",
                "extra_metadata": {},
            }

            for index, chunk_text in enumerate(chunks, start=1):
                logger.info(
                    "Extracting chunk %d/%d for %s",
                    index,
                    len(chunks),
                    filename,
                )

                chunk_payload = self._generate_json_for_chunk(
                    chunk_text=chunk_text,
                    filename=filename,
                )

                merged_payload = self._merge_json_objects(
                    merged_payload,
                    chunk_payload,
                )

            # These fields always come from the original document.
            merged_payload["filename"] = filename
            merged_payload["content"] = content

            return DocumentExtraction.model_validate(
                merged_payload
            )

        except Exception:
            logger.exception(
                "LLM extraction failed for %s",
                filename,
            )
            return None


    def _existing_extraction(
            self,
            existing: Any,
        ) -> DocumentExtraction:
            metadata = getattr(existing, "doc_metadata", None) or {}
    
            # Support repositories returning either ORM documents or
            # dictionaries with a "metadata" field.
            if not metadata and isinstance(existing, dict):
                metadata = existing.get("metadata", {})
    
            metadata = dict(metadata)
            doc_type = metadata.pop("doc_type", "other")
    
            return DocumentExtraction.model_validate(
                {
                    "filename": existing.filename,
                    "content": existing.content,
                    "doc_type": doc_type,
                    "extra_metadata": metadata,
                }
            )
        
    def process_document(
            self,
            path: str | Path,
        ) -> ExtractionResult:
            path = Path(path)
    
            # 1. Check for an existing document before expensive API calls.
            with transaction() as db:
                repo = DocumentRepository(db)
                existing = repo.get_by_filename(path.name)
    
                if existing is not None:
                    logger.info(
                        "Skip duplicate: %s (id=%d)",
                        path.name,
                        existing.id,
                    )
                    return ExtractionResult(
                        success=True,
                        document=self._existing_extraction(existing),
                    )
    
            # 2. Load the original document.
            content = self.load_document(path)
    
            if content is None:
                return ExtractionResult(
                    success=False,
                    error=f"load failed: {path.name}",
                )
    
            # 3. Extract and validate the consolidated metadata.
            extraction = self.extract_metadata(
                content,
                filename=path.name,
            )
    
            if extraction is None:
                return ExtractionResult(
                    success=False,
                    error=f"extract failed: {path.name}",
                )
    
            # 4. Split and embed exactly the chunks we intend to persist.
            chunks = self.split_content(content)
    
            try:
                embeddings = self.generate_embeddings(chunks)
            except Exception:
                logger.exception(
                    "Embedding generation failed for %s",
                    path.name,
                )
                return ExtractionResult(
                    success=False,
                    error=f"embedding failed: {path.name}",
                )
    
            if len(chunks) != len(embeddings):
                return ExtractionResult(
                    success=False,
                    error=f"chunk/embedding count mismatch: {path.name}",
                )
    
            # 5. Persist parent and children in one transaction.
            with transaction() as db:
                repo = DocumentRepository(db)
                chunk_repo = ChunkRepository(db)
    
                # Recheck to reduce the chance of concurrent duplicate inserts.
                existing = repo.get_by_filename(path.name)
    
                if existing is not None:
                    logger.info(
                        "Skip concurrent duplicate: %s",
                        path.name,
                    )
                    return ExtractionResult(
                        success=True,
                        document=self._existing_extraction(existing),
                    )
    
                db_data = extraction.to_db_format()    
                doc = repo.create(**db_data)
    
                # Ensure doc.id is available before inserting child rows.
                db.flush()
    
                chunk_repo.create_chunks_batch(
                    [
                        {
                            "document_id": doc.id,
                            "content": chunk_text,
                            "chunk_index": index,
                            "embedding": embedding,
                        }
                        for index, (chunk_text, embedding) in enumerate(
                            zip(chunks, embeddings),
                            start=0,
                        )
                    ]
                )
    
                logger.info(
                    "Saved %s (id=%d, chunks=%d)",
                    path.name,
                    doc.id,
                    len(chunks),
                )
    
                return ExtractionResult(
                    success=True,
                    document=extraction,
                )

    def process_batch(self, paths: list[str | Path]) -> list[ExtractionResult]:
        results: list[ExtractionResult] = []
        for path in paths:
            try:
                results.append(self.process_document(path))
            except Exception as e:
                # safety net — eroare neașteptată nu oprește batch-ul
                logger.exception("Unexpected error on %s", path)
                results.append(ExtractionResult(success=False, error=str(e)))
        return results

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

    agent = DbAgent(model=MODEL)
    results = agent.process_batch(["sample_data/facturi/docx/factura004.docx"])

    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    logger.info("Batch done: %d ok, %d failed", len(successful), len(failed))
    for r in failed:
        logger.error("Failed: %s", r.error)


if __name__ == "__main__":
    main()