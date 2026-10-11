import json
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
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


logger = logging.getLogger(__name__)

load_dotenv()

BASE_URL = os.getenv(
    "LLM_BASE_URL",
    "http://localhost:4000/v1",
)
MODEL = os.getenv("LLM_MODEL", "llama")
API_KEY = os.getenv("LITELLM_API_KEY", "local")

EXTRACTION_PROMPT = """Extract structured metadata from the document below.

Filename: {filename}

Document content:
{content}

Return a JSON object matching this schema:
{schema}

Rules:
- Return only valid JSON.
- Do not include Markdown code fences.
- Do not invent missing information.
- Use null for missing information when allowed by the schema.
- Preserve names, dates, amounts, and identifiers exactly as written.
- Extract only information supported by the document.
"""


class DbAgent:
    def __init__(
        self,
        model: str = MODEL,
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        ),
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_threshold: int = 4000,
        base_url: str = BASE_URL,
        api_key: str = API_KEY,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.embedding_model_name = embedding_model

        # Lazy initialization
        self._client: OpenAI | None = None
        self._embedding_model: SentenceTransformer | None = None

        self.chunk_threshold = chunk_threshold

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    @property
    def client(self) -> OpenAI:
        """Create the OpenAI-compatible LiteLLM client lazily."""
        if self._client is None:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=120.0,
                max_retries=2,
            )
        return self._client

    @property
    def embeddings_model(self) -> SentenceTransformer:
        """Load the local embedding model lazily."""
        if self._embedding_model is None:
            self._embedding_model = SentenceTransformer(
                self.embedding_model_name
            )
        return self._embedding_model

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
        """Generate normalized embeddings for document chunks."""
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
        """Extract and validate structured metadata using LiteLLM."""
        schema = DocumentChunkExtraction.model_json_schema()

        prompt = EXTRACTION_PROMPT.format(
            filename=filename,
            content=chunk_text,
            schema=json.dumps(schema, ensure_ascii=False),
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured metadata from legal "
                        "contracts and invoices. Follow the supplied "
                        "JSON schema. Never invent missing information."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            temperature=0,
            max_tokens=2048,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError(
                f"Empty LLM response for {filename}"
            )

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON returned for {filename}: {exc}"
            ) from exc

        if not isinstance(parsed, dict):
            raise TypeError(
                f"Expected JSON object for {filename}, "
                f"got {type(parsed).__name__}"
            )

        validated = DocumentChunkExtraction.model_validate(parsed)
        return validated.model_dump()

    def _merge_json_objects(
        self,
        base: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge extracted metadata while preserving existing values."""
        merged = dict(base)

        for key, value in incoming.items():
            if value is None:
                continue

            if key not in merged or merged[key] is None:
                merged[key] = value
                continue

            current = merged[key]

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

        return merged

    def load_document(self, path: str | Path) -> str | None:
        """Load PDF, DOCX, or TXT content."""
        path = Path(path)

        try:
            match path.suffix.lower():
                case ".pdf":
                    loader = PyPDFLoader(str(path))
                case ".docx":
                    loader = Docx2txtLoader(str(path))
                case ".txt":
                    loader = TextLoader(
                        str(path),
                        autodetect_encoding=True,
                    )
                case _:
                    logger.warning(
                        "Unsupported format: %s",
                        path.suffix,
                    )
                    return None

            docs = loader.load()
            return "\n\n".join(
                document.page_content for document in docs
            )

        except Exception:
            logger.exception("Failed to load %s", path)
            return None

    def extract_metadata(
        self,
        content: str,
        filename: str,
    ) -> DocumentExtraction | None:
        """Extract metadata from all chunks and merge the results."""
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

            # Infer the document type if the LLM did not provide a valid one.
            doc_type = merged_payload.get("doc_type")

            if doc_type not in {"invoice", "contract"}:
                filename_lower = filename.lower()

            if "factur" in filename_lower or "invoice" in filename_lower:
                doc_type = "invoice"
            elif "contract" in filename_lower:
                doc_type = "contract"
            else:
                logger.error(
                    "Could not determine document type for %s",
                    filename,
                )
                return None

            merged_payload["doc_type"] = doc_type
            merged_payload["filename"] = filename
            merged_payload["content"] = content

            return DocumentExtraction.model_validate(merged_payload)

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
        """Convert an existing database record to the extraction schema."""
        metadata = getattr(existing, "doc_metadata", None) or {}

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
        """Load, extract, embed, and persist one document."""
        path = Path(path)

        # 1. Skip documents that already exist.
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

        # 2. Load document content.
        content = self.load_document(path)

        if content is None:
            return ExtractionResult(
                success=False,
                error=f"load failed: {path.name}",
            )

        # 3. Extract structured metadata.
        extraction = self.extract_metadata(
            content,
            filename=path.name,
        )

        if extraction is None:
            return ExtractionResult(
                success=False,
                error=f"extract failed: {path.name}",
            )

        # 4. Generate embeddings for the same chunks we will persist.
        chunks = self.split_content(content)

        try:
            embeddings = self.generate_embeddings(chunks)
        except Exception as exc:
            logger.exception(
                "Embedding generation failed for %s",
                path.name,
            )
            return ExtractionResult(
                success=False,
                error=f"embedding failed: {path.name}: {exc}",
            )

        if len(chunks) != len(embeddings):
            return ExtractionResult(
                success=False,
                error=f"chunk/embedding count mismatch: {path.name}",
            )

        # 5. Persist the document and its chunks in one transaction.
        with transaction() as db:
            repo = DocumentRepository(db)
            chunk_repo = ChunkRepository(db)

            # Recheck to reduce concurrent duplicate inserts.
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

    def process_batch(
        self,
        paths: list[str | Path],
    ) -> list[ExtractionResult]:
        """Process multiple documents without stopping on one failure."""
        results: list[ExtractionResult] = []

        for path in paths:
            try:
                results.append(self.process_document(path))
            except Exception as exc:
                logger.exception(
                    "Unexpected error on %s",
                    path,
                )
                results.append(
                    ExtractionResult(
                        success=False,
                        error=str(exc),
                    )
                )

        return results


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

    from transformers.utils import logging as transformers_logging
    transformers_logging.disable_progress_bar()

    agent = DbAgent(model=MODEL)

    results = agent.process_batch(
        ["sample_data/contracte/pdf/ContractImobil.pdf"]
    )

    successful = [result for result in results if result.success]
    failed = [result for result in results if not result.success]

    logger.info(
        "Batch done: %d ok, %d failed",
        len(successful),
        len(failed),
    )

    for result in failed:
        logger.error("Failed: %s", result.error)


if __name__ == "__main__":
    main()