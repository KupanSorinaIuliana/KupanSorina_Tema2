from dataclasses import dataclass


@dataclass
class PromptTemplate:
    name: str
    version: str
    description: str
    prompt: str
