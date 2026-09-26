# PromptRegistry (load YAML + Jinja2)
import re
from pathlib import Path

import yaml
from jinja2 import Template

from .template import PromptTemplate


class PromptRegistry:
    def __init__(self, folder: str):
        self._templates = self._load(folder)

    def _load(self, folder: str) -> dict[str, PromptTemplate]:
        # Accept both .yaml and .yml files so the project can use either shape.
        templates: dict[str, PromptTemplate] = {}
        root = Path(folder)
        for pattern in ("*.yaml", "*.yml"):
            for path in root.rglob(pattern):
                data = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not data:
                    continue
                tpl = PromptTemplate(**data)
                templates[tpl.name] = tpl
        return templates

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {
            token.lower()
            for token in re.findall(r"[a-zA-ZăîșțĂÎȘȚ0-9]+", text or "")
        }

    def select_prompt_name(self, message: str) -> str:
        """Choose the best prompt template based on words that appear in the prompt metadata."""
        message_tokens = self._tokenize(message)
        if not message_tokens:
            return "ask_analyst"

        best_name = "ask_analyst"
        best_score = -1

        for name, template in self._templates.items():
            prompt_text = " ".join([
                template.name,
                template.description,
                template.prompt,
            ]).lower()

            score = sum(1 for token in message_tokens if token in prompt_text)
            if score > best_score:
                best_score = score
                best_name = name

        return best_name

    def render(self, name: str, **variabile) -> str:
        template = self._templates[name]
        return Template(template.prompt).render(**variabile)
