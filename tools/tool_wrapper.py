# ToolWrapper.call() + ToolWrapper.catalog()
import json
from tools.registry import TOOL_REGISTRY

# For the agent to execute the chosen tool
class ToolWrapper:
    @staticmethod
    def call(name: str, args: dict) -> str:
        # Lookup în registry
        if name not in TOOL_REGISTRY:
            return f"Eroare: tool '{name}' nu există."

        tool = TOOL_REGISTRY[name]

        # Validate — Pydantic verifies types and constraints
        try:
            params = tool["params_model"](**args)
        except Exception as e:
            return f"Eroare validare pentru '{name}': {e}"

        # Execute + Return
        try:
            result = tool["func"](params)
            if isinstance(result, (dict, list, tuple, set)):
                return json.dumps(result, ensure_ascii=False, indent=2, default=str)
            return str(result)
        except Exception as e:
            return f"Eroare execuție '{name}': {e}"

    # What tools exists and what arguments they take, for the LLM to know the tools available
    @staticmethod
    def catalog() -> list[dict]:
        """Return a provider-agnostic tool catalog.

        We include both the generic OpenAI-compatible function schema and the
        older "name + input_schema" form so different model providers can consume
        the same registry without requiring custom catalog code in each agent.
        """
        catalog = []
        for name, tool in TOOL_REGISTRY.items():
            schema = tool["params_model"].model_json_schema()
            catalog.append(
                {
                    "name": name,
                    "description": tool["description"],
                    "input_schema": schema,
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool["description"],
                        "parameters": schema,
                    },
                }
            )
        return catalog
