from datetime import datetime, timezone
import ast
import html
from tools.extraction_pipeline import ExtractionPipeline
from tools.params_models import CalculatorParams, ExtractContractParams, WebSearchParams, GetDatetimeParams, ExtractInvoiceParams, BatchExtractParams
from tools.registry import register_tool
from tools.document_schemas import Invoice, Contract
from tools.save_json import save_json, run_save_pipeline
from ddgs import DDGS


@register_tool(
    params_model=WebSearchParams,
    description=
    """
    Caută pe web și returnează rezultate relevante cu titlu, URL și descriere.
    Exemple de utilizare:
    "caută cele mai recente știri despre legea contractelor"
    "gaseste informații despre șablonul de contract de vânzare a unei mașini"
    """,
)
def web_search(params: WebSearchParams) -> str:
    if not params.query or not params.query.strip():
        return "Eroare: the query can not be empty."

    with DDGS() as ddgs:
        results = ddgs.text(params.query.strip(), max_results=params.max_results)
        return [
            {
                "title": html.unescape(item.get("title")),
                "url": item.get("href"),
                "snippet": item.get("body"),
            } 
            for item in results
        ]


@register_tool(
    params_model=CalculatorParams,
    description=
    """
        Evalueaza o expresie matematică și returnează rezultatul.
        Exemple de utilizare:
        "calc 2 + 3 * 4"
        "calculeaza 10 / 2"
        "rezolva 5^2 + 3"
        " 2 + 3 * (4 - 1) "
    """,
)
def calculator(params: CalculatorParams) -> str:
    try:
        node = ast.parse(params.expression, mode="eval")
        value = eval(compile(node, "", "eval"))
        return f"{params.expression} = {str(value)}"
    except Exception as exc:  # pragma: no cover - simple fallback
        return f"Eroare calcul: {exc}"


@register_tool(
    params_model=GetDatetimeParams,
    description=
     """
     Preia data și ora curentă locală și returnează-o ca un șir de caractere în format ISO 8601.

     Exemple de utilizare:
        "care este data și ora curentă?"
        "care este data curentă?"
        "ce zi este astăzi?"
        "care este ora curentă?"
        """,
)
def get_datetime(params: GetDatetimeParams) -> str:
    return f"Data si ora curenta: {datetime.now(timezone.utc).isoformat()}"

@register_tool(
    params_model=ExtractInvoiceParams,
    description=
    """
        Extrage date dintr-o factura (numar, data, client, total, produse).
        Exemple de utilizare:
        "extrage date din aceasta factura"
        "parseaza aceasta factura"
    """,
)
def extract_invoice(params: ExtractInvoiceParams) -> dict:
    """Extrage date structurate dintr-o factură (numar, data, client, total, produse).
    Folosește când userul cere parsare/extracție dintr-o factură."""
    try:
        pipeline = ExtractionPipeline()
        result = pipeline.process(params.file_path, Invoice)
        save_json(result, "facturi")
        return result.model_dump()
    except FileNotFoundError:
        return {"error": f"Fișier inexistent: {params.file_path}"}
    except Exception as e:
        return {"error": f"Extracție eșuată: {str(e)}"}

@register_tool(
    params_model=ExtractContractParams,
    description=
    """
        Extrage date structurate dintr-un contract (numar, data incheiere, prestator, beneficiar, valoare, durata luni, obligatii prestator).
        Exemple de utilizare:
        "extrage date din acest contract"
        "parseaza acest document contract"
    """,
)
def extract_contract(params: ExtractContractParams) -> dict:
    """Extrage date structurate dintr-un contract (numar, data incheiere, prestator, beneficiar, valoare, durata luni, obligatii prestator).
    Folosește când userul cere parsare/extracție dintr-un contract."""
    try:
        pipeline = ExtractionPipeline()
        result = pipeline.process(params.file_path, Contract)
        save_json(result, "contracte")
        return result.model_dump()
    except FileNotFoundError:
        return {"error": f"Fișier inexistent: {params.file_path}"}
    except Exception as e:
        return {"error": f"Extracție eșuată: {str(e)}"}

@register_tool(
    params_model=BatchExtractParams,
    description=
    """
    Extract and save multiple files in parallel using the document registry and worker queue.
    Usage examples:
    "process these invoice files"
    "save all contract documents from this folder"
    """,
)
def batch_extract_documents(params: BatchExtractParams) -> dict:
    """Procesare batch de fișiere folosind pipeline-ul de salvare în thread-uri."""
    try:
        items = [(file_path, params.doc_type) for file_path in params.file_paths]
        run_save_pipeline(items, num_workers=params.num_workers)
        return {
            "doc_type": params.doc_type,
            "processed_files": len(params.file_paths),
            "num_workers": params.num_workers,
            "status": "completed",
        }
    except Exception as exc:
        return {"error": f"Procesare batch eșuată: {str(exc)}"}