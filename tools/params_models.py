# Pydantic BaseModel pentru fiecare tool
from pydantic import BaseModel, Field

class CalculatorParams(BaseModel):
    expression: str = Field(
        description="The mathematical expression to evaluate (e.g., '2 + 3 * 4')",
        min_length=1
    )

class WebSearchParams(BaseModel):
    query: str = Field(
        description=" The search terms for web search",
        min_length=2
    )
    max_results: int = Field(
        default=5,
        description="Maximum number of results to return",
        ge=1, le=20
    )

class GetDatetimeParams(BaseModel):
    pass  # There are no parameters for this tool, but we define a class for consistency.

class ExtractInvoiceParams(BaseModel):
    """Parametri pentru tool-ul extract_invoice."""
    file_path: str = Field(
        description="Calea către fișier (PDF, DOCX, TXT)",
        min_length=1,
    )

class ExtractContractParams(BaseModel):
    """Parametri pentru tool-ul extract_contract."""
    file_path: str = Field(
        description="Calea către fișier (PDF, DOCX, TXT)",
        min_length=1,
    )


class BatchExtractParams(BaseModel):
    """Parametri pentru procesarea în bloc a mai multor documente."""
    file_paths: list[str] = Field(
        description="Lista de căi către fișierele de procesat",
        min_length=1,
    )
    doc_type: str = Field(
        description="Tipul documentului: 'factura' sau 'contract'",
        min_length=1,
    )
    num_workers: int = Field(
        default=4,
        description="Numărul de workeri pentru procesarea paralelă",
        ge=1,
        le=20,
    )

class SearchParams(BaseModel):
    query: str = Field(
        description=" Termenii de căutare pentru căutarea documentelor",
        min_length=2
    )