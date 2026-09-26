# Project overview

This project contains a small Python agent scaffold with:

- a prompt registry that loads YAML templates from the `prompts` folder
- a tool registry and wrapper for callable tools
- a simple conversational `QAAgent` wrapper around a LangChain chat model

## Structure

- `agent.py` – conversational agent wrapper
- `prompts/` – YAML prompt templates and registry loader
- `tools/` – tool definitions, parameter schemas, and registry

## Quick usage

```python
from tools import ToolWrapper

print(ToolWrapper.call("calculator", {"expression": "2 + 3 * 4"}))
print(ToolWrapper.call("get_datetime", {"query": "Bucharest"}))
```

```python
from prompts import PromptRegistry

registry = PromptRegistry("prompts")
print(registry.render("ask_analyst", rol="expert", domeniu="programming", max_cuvinte=50))
```

## Notes

- The tool calculator uses a restricted safe evaluator instead of raw `eval`.
- The prompt registry reads all `.yaml` files in the `prompts` directory and renders Jinja2 templates.


## Project setup

- Run the following command in the terminal:
    python -m pip install langchain-openai jinja2 pyyaml pydantic httpx
- Start docker:
    docker compose up -d
- Check that LiteLLM is alive:
    curl http://localhost:4000/health/liveliness
- Pull the model you need from the terminal:
    ollama pull qwen2.5:3b
- Run the agent:
    python agent.py