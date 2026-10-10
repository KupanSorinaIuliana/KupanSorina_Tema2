import json
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Type
from google import genai
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from tools.extraction_layer.loader import load_document


class ExtractionPipeline:
    """Pipeline complet de extracție cu validare înapoi la schema Pydantic."""

    def __init__(self, model: str = "gemini-3.8-flash"):
        self.model = model
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        self._client: genai.Client | None = None

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    @staticmethod
    def should_chunk(documents: list[Document], max_size: int = 4000) -> bool:
        text = "\n".join((doc.page_content or "") for doc in documents)
        return len(text) > max_size

    @staticmethod
    def _strip_markdown_fence(raw_text: str) -> str:
        text = raw_text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        return text.strip()

    @staticmethod
    def _merge_json_objects(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in incoming.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ExtractionPipeline._merge_json_objects(merged[key], value)
            elif key in merged and isinstance(merged[key], list) and isinstance(value, list):
                merged[key] = merged[key] + value
            else:
                merged[key] = value
        return merged

    @staticmethod
    def _parse_response_payload(raw_response: Any) -> dict[str, Any]:
        if hasattr(raw_response, "text"):
            payload = raw_response.text
        elif isinstance(raw_response, str):
            payload = raw_response
        else:
            payload = str(raw_response)

        payload = ExtractionPipeline._strip_markdown_fence(payload)
        parsed = json.loads(payload)

        if isinstance(parsed, list):
            if not parsed:
                raise ValueError("Extracția LLM a returnat o listă goală.")
            parsed = parsed[0]

        if not isinstance(parsed, dict):
            raise ValueError(f"Datele extrase nu sunt un obiect JSON valid: {type(parsed)!r}")

        return parsed

    def _generate_json_for_chunk(self, chunk_text: str, schema: Type[BaseModel]) -> dict[str, Any]:
        raw_response = self.client.models.generate_content(
            model=self.model,
            contents=chunk_text,
            config={
                "response_mime_type": "application/json",
                "response_json_schema": schema.model_json_schema(),  # Gemini → Pydantic direct
            },
        )
        return self._parse_response_payload(raw_response)

    def process(self, file_path: str, schema: Type[BaseModel]) -> BaseModel:
        docs = load_document(file_path)
        chunks = self.splitter.split_documents(docs) if self.should_chunk(docs) else docs

        extracted_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            extracted_chunks.append(self._generate_json_for_chunk(chunk.page_content, schema))

        merged_payload: dict[str, Any] = {}
        for chunk_payload in extracted_chunks:
            merged_payload = self._merge_json_objects(merged_payload, chunk_payload)

        return schema.model_validate(merged_payload)


def _process_one(file_path: str, schema: Type[BaseModel]) -> BaseModel:
    """Helper la nivel de modul necesar pentru multiprocessing."""
    pipeline = ExtractionPipeline()
    return pipeline.process(file_path, schema)


def batch_extract(files: list[str], schema: Type[BaseModel], num_workers: int = 4):
    """Extrage date pentru mai multe fișiere în paralel și returnează rezultatele."""
    results = {"success": [], "errors": []}

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {
            executor.submit(_process_one, file_path, schema): file_path
            for file_path in files
        }

        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                data = future.result()
                results["success"].append({"file": file_path, "data": data})
            except Exception as exc:
                results["errors"].append({"file": file_path, "error": str(exc)})

    return results