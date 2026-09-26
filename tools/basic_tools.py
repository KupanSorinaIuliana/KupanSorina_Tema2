from datetime import datetime, timezone
import ast
import html
import operator
from tools.params_models import CalculatorParams, WebSearchParams, GetDatetimeParams
from tools.registry import register_tool
from ddgs import DDGS


@register_tool(
    params_model=WebSearchParams,
    description="Cauta pe web si returneaza rezultate relevante, cu titlu, URL si descriere.",
)
def web_search(params: WebSearchParams) -> str:
    """Cauta pe web si returneaza rezultatele relevante ca text."""
    if not params.query or not params.query.strip():
        return "Eroare: query-ul nu poate fi gol."

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
        Evalueaza o expresie matematica.
        Exemplu de folosire:
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
        Get the current local date and time and return it as an an ISO 8601 date time string.

        Exemplu de folosire:
        "ce data e azi?"
        "ce zi e azi?"
        "ce ora e acum?"
        "time stamp asta"
        """,
)
def get_datetime(params: GetDatetimeParams) -> str:
    return f"Data si ora curenta: {datetime.now(timezone.utc).isoformat()}"
