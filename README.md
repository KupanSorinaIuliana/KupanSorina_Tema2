# Project overview

This project contains a QA agent built using the following standards:

- a prompt registry that loads YAML templates from the `prompts` folder
- a tool registry and wrapper for callable tools
- a simple conversational `QAAgent` wrapper around a LangChain chat model

## Structure

- `agent.py` – conversational agent wrapper
- `prompts/` – YAML prompt templates and registry loader
- `tools/` – tool definitions, parameter schemas, and registry
- `docker-compose.yml` – local container stack that starts the LiteLLM proxy and any supporting services
- `litellm-config.yaml` – LiteLLM configuration for routing requests to the local Ollama model backend

## Project setup

- Install the needed imports if they are not already installed.
- Start docker:
    docker compose up -d
- Check that LiteLLM is alive:
    curl http://localhost:4000/health/liveliness
- Pull the model you need from the terminal (the model can be changed in the agent.py file):
    ollama pull llama3.2:3b
- Run the agent:
    python agent.py