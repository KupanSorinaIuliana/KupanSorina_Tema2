# TOOL_REGISTRY + register_tool
## here we register the tools and handle them
from typing import Any, Callable

from pydantic import BaseModel

# Registry global pentru tool-uri
TOOL_REGISTRY: dict[str, dict[str, Any]] = {}


def register_tool(*, params_model: type[BaseModel], description: str | None = None):
    """Register a tool with its parameter model and metadata."""

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        TOOL_REGISTRY[func.__name__] = {
            "func": func,
            "params_model": params_model,
            "description": description or func.__doc__ or "",
        }
        return func

    return decorator
