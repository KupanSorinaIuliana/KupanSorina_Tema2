from datetime import datetime, timezone
import ast
import html
from tools.params_models import CalculatorParams, WebSearchParams, GetDatetimeParams
from tools.registry import register_tool
from ddgs import DDGS


@register_tool(
    params_model=WebSearchParams,
    description=
    """
    Search the web and return relevant results with title, URL, and description.
    Usage examples:
    "search for the latest news on contract law"
    "find information about the Eiffel Tower"

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
        Evaluate a mathematical expression and return the result.
        Usage examples:
        "calc 2 + 3 * 4"
        "calculate 10 / 2"
        "solve 5^2 + 3"
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

        Usage examples:
        "what is the current date and time?"
        "what is the current date?"
        "what day is it today?"
        "what time is it now?"
        "time stamp this"
        """,
)
def get_datetime(params: GetDatetimeParams) -> str:
    return f"Data si ora curenta: {datetime.now(timezone.utc).isoformat()}"
