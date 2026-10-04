import json
from pathlib import Path
from queue import Queue
from threading import Thread
from pydantic import BaseModel
from tools.document_schemas import Contract, Invoice
from tools.extraction_pipeline import ExtractionPipeline

STOP = None   # sentinel propagat între workeri

EXTRACTION_REGISTRY = {
    "factura": {
        "schema": Invoice,
        "chunk_size": None,
    },
    "contract": {
        "schema": Contract,
        "chunk_size": 2000,
    },
}


def save_json(data: BaseModel, doc_type: str):
    out = Path("extracted_data") / doc_type
    out.mkdir(parents=True, exist_ok=True)

    if not hasattr(data, "numar"):
        filename = f"document_{abs(hash(str(data)))}.json"
    else:
        filename = f"{str(data.numar).replace('/', '_')}.json"

    with open(out / filename, "w", encoding="utf-8") as f:
        json.dump(data.model_dump(), f, indent=2, ensure_ascii=False)


def log_to_db(data: BaseModel, doc_type: str):
    """Hook opțional pentru persistenta suplimentară."""
    return None


def notify(data: BaseModel, doc_type: str):
    """Hook opțional pentru notificarile ulterioare."""
    return None


def extract_document(file_path: str, doc_type: str):
    """Extrage un document folosind registry-ul de tipuri."""
    config = EXTRACTION_REGISTRY[doc_type]
    pipeline = ExtractionPipeline()
    return pipeline.process(file_path, config["schema"])


def run_save_pipeline(files_to_process, num_workers: int = 5):
    """Pornește workeri, procesează fișiere, oprește curat la final."""
    queue: Queue = Queue(maxsize=100)

    def worker(worker_id: int):
        while True:
            item = queue.get()
            if item is STOP:
                queue.put(STOP)
                break
            data, doc_type = item
            try:
                save_json(data, doc_type)
                log_to_db(data, doc_type)
                notify(data, doc_type)
            except Exception as exc:
                print(f"Worker {worker_id} error: {exc}")

    workers = [Thread(target=worker, args=(i,)) for i in range(num_workers)]
    for worker in workers:
        worker.start()

    for file_path, doc_type in files_to_process:
        result = extract_document(file_path, doc_type)
        queue.put((result, doc_type))

    queue.put(STOP)
    for worker in workers:
        worker.join()
