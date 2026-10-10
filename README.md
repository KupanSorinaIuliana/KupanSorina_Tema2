# Project overview
- `agents` - containing all the existing agents
- `db_layer` - repositories, rag service, models, index creation
- `extracted_data` - json extracted documents
- `migrations` - document and document chunk migrations
- `prompts` - existing prompts to be used by the chat agents
- `sample_data` - invoices and contracts for the extracting agents
- `tools` - existing tools infrastructure + json extraction helpers
- `env` - add you gemini api key in here and change the model if needed, same for the database url

## Agents folder
-` __init__.py`
- `initial_agent.py` – conversational agent wrapper
- `json_agent.py` - extractor agent from docx, pdf, and txt files (use files from the sample_data folder) to json files (output in extracted_data folder)
- `db_agent.py` - extractor from docx, pdf, and txt files (use files from the sample_data folder) to a db format (including db save with chunks and embeddings)
- `analyst_agent.py` - responds based on the db existing data using the similarity search tool

## Database layer
- `__init__.py`
- `chunk_repository` - crud operations for the DocumentChunk entity
- `database` - creates a local database session and sets the database url
- `document_extractor` - the transformer betweeen the extraction and the db entities
- `document_repository` - crud operations for the Document entity
- `exceptions` - custom exceptions
- `index_creation` - create index scripts for the IVFFLAT index
- `models` - document and document chunk definitions
- `rag_service` - search methods using the sentence transformer
- `transaction_manager` - starts a new db session, commits the data, performs rollback if needed, and closes the database

## Prompts – YAML prompt templates and registry loader
- `__init__.py`
- `analyst.yaml` - analyst prompt
- `extract.yaml` - extract prompt
- `planner.yaml` - planner prompt
- `registry.py` - register the prompts, render the best one based on similarity of the text
- `summary.yaml` - summary prompts
- `template.py` - the template of the prompt formats

## Tools
- `__init__.py`
- `basic_tools.py` - tool definitions
- `params_models.py` - tool parameters
- `registry.py` - registering all the existing tools
- `tool_wrapper.py` - tool rendering
- `extraction_layer`
    - `document_schemas.py` - contract and invoice schema models
    - `extraction_pipeline.py` - extraction process using gemini
    - `loader.py` - loads the document using the loader registry
    - `save_json.py` - calls the extraction pipeline and saves the json

## Project setup
- Install the needed imports if they are not already installed.
- Start docker:
    docker compose up -d
- Add you gemini key, model and database url in the env file
- Run the database migrations:
    alembic upgrade head
- Run the index creation file from the database layer folder:
    python index_creation.py
- Run the agent (would be good to run the db agent to have some data in the database before testing analyst and intial agents):
    python -m agents.db_agent
    python -m agents.json_agent
    python -m agents.analyst_agent
    python -m agents.initial_agent