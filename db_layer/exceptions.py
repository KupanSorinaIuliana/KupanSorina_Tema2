class DocumentError(Exception):
   """Bază pentru toate erorile din domeniul Document."""


class DocumentNotFoundError(DocumentError):
    def __init__(self, doc_id: int) -> None:
        super().__init__(f"Document with id={doc_id} not found")
        self.doc_id = doc_id


class DuplicateDocumentError(DocumentError):
    def __init__(self, filename: str) -> None:
        super().__init__(f"Document with filename='{filename}' already exists")
        self.filename = filename


class InvalidMetadataError(DocumentError):
    def __init__(self, reason: str) -> None:
        super().__init__(f"Invalid metadata: {reason}")
        self.reason = reason