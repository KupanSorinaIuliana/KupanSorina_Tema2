# QA agent / orchestrare LLM
from __future__ import annotations

import json
import re
import uuid
from typing import Any, Iterator
import asyncio
import httpx

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama

from prompts import PromptRegistry
from tools import basic_tools  # noqa: F401 - ensure tool registry is populated
from tools.tool_wrapper import ToolWrapper

BASE_URL = "http://localhost:4000"
MODEL = "llama3.2"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Be concise. "
    "Use tools when needed, especially the calculator tool for arithmetic."
)
MESSAGE = "Calculate 10*2+20."


def build_llm() -> BaseChatModel:
    return ChatOllama(
        model=MODEL,
        temperature=0.7,
        max_tokens=1024,
    ).bind_tools(ToolWrapper.catalog())

class QAAgent:
    """Agent conversational cu istoric - un wrapper peste un ChatModel."""

    def __init__(self, llm: BaseChatModel, system_prompt: str, max_iterations: int = 3):
        self.llm: BaseChatModel = llm
        self.system_prompt: str = system_prompt
        self.max_iterations = max_iterations
        self.prompt_registry = PromptRegistry("prompts")
        self.history: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    def build_system_prompt(self, message: str, rol: str = "expert", domeniu: str = "general") -> str:
        """Render a task-specific system prompt from the prompt YAML files."""
        prompt_name = self.prompt_registry.select_prompt_name(message)
        render_kwargs = {"rol": rol, "domeniu": domeniu}

        if prompt_name in {"ask_planner", "ask_summary"}:
            render_kwargs["max_cuvinte"] = 80
        elif prompt_name == "ask_analyst":
            render_kwargs["max_cuvinte"] = 120

        return self.prompt_registry.render(prompt_name, **render_kwargs)

    def _build_messages(self, message: str) -> list[BaseMessage]:
        selected_system_prompt = self.build_system_prompt(message)
        self.system_prompt = selected_system_prompt
        non_system_history = [msg for msg in self.history if not isinstance(msg, SystemMessage)]
        self.history = [SystemMessage(content=selected_system_prompt)] + non_system_history
        return self.history + [HumanMessage(content=message)]

    def _normalize_tool_call(self, tool_call: dict[str, Any]) -> dict[str, Any]:
        """Normalize different provider-specific tool-call payloads to one shape."""
        if not isinstance(tool_call, dict):
            return {"name": None, "arguments": {}, "id": None}

        func = tool_call.get("function") if isinstance(tool_call.get("function"), dict) else {}
        name = (
            tool_call.get("name")
            or tool_call.get("tool_name")
            or func.get("name")
        )
        arguments = (
            tool_call.get("args")
            or tool_call.get("arguments")
            or tool_call.get("tool_args")
            or func.get("arguments")
        )

        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if arguments is None:
            arguments = {}

        return {
            "id": tool_call.get("id") or tool_call.get("call_id") or f"call_{uuid.uuid4().hex[:8]}",
            "name": name,
            "arguments": arguments,
        }

    def execute_tool(self, tool_call: dict[str, Any]) -> str:
        normalized = self._normalize_tool_call(tool_call)
        name = normalized["name"]
        args = normalized["arguments"]

        if not name:
            raise ValueError(f"Tool call fără nume: {tool_call!r}")

        return ToolWrapper.call(name, args or {})

    @staticmethod
    def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
        tool_calls = getattr(response, "tool_calls", None)
        if not tool_calls:
            extra = getattr(response, "additional_kwargs", {}) or {}
            tool_calls = extra.get("tool_calls")
        return tool_calls or []

    @staticmethod
    def _summarize_tool_results(tool_results: list[tuple[str, str]]) -> str:
        if not tool_results:
            return ""

        summary_parts = []
        for name, result in tool_results:
            summary_parts.append(result)

        return "\n".join(summary_parts)

    def react_loop(self, llm, mesaje: list, max_iterations: int = 5) -> str:
        """Loop Think → Act → Observe."""
        tool_results: list[tuple[str, str]] = []
        seen_tool_names: set[str] = set()

        for _ in range(max_iterations):
            raspuns = llm.invoke(mesaje)
            mesaje.append(raspuns)

            tool_calls = self._extract_tool_calls(raspuns)

            if not tool_calls:
                content = str(getattr(raspuns, "content", "") or "")
                if content and content.strip().startswith('{'):
                    return self._summarize_tool_results(tool_results) or content
                return content

            tool_names_in_response = []
            for tool_call in tool_calls:
                normalized = self._normalize_tool_call(tool_call)
                tool_name = normalized.get("name")
                if tool_name:
                    tool_names_in_response.append(tool_name)

                if tool_name and tool_name in seen_tool_names:
                    if tool_results:
                        return self._summarize_tool_results(tool_results)
                    break

                rezultat = self.execute_tool(normalized)
                tool_call_id = normalized.get("id") or f"call_{uuid.uuid4().hex[:8]}"
                tool_name = normalized.get("name")
                if tool_name:
                    seen_tool_names.add(tool_name)
                tool_results.append((tool_name or "tool", str(rezultat)))
                mesaje.append(ToolMessage(
                    content=str(rezultat),
                    tool_call_id=tool_call_id,
                    name=tool_name,
                ))

        if tool_results:
            return self._summarize_tool_results(tool_results)

        raise RuntimeError("Max iterations atins fără răspuns final")

    def chat(self, message: str) -> str:
        """Trimite mesaj, primeste raspuns complet, actualizeaza istoricul."""
        messages = self._build_messages(message)
        answer = self.react_loop(self.llm, messages)
        self.history.append(HumanMessage(content=message))
        self.history.append(AIMessage(content=answer))
        return answer

    def stream(self, message: str) -> Iterator[str]:
        """Yield chunks pe masura ce vin de la LLM."""
        messages = self._build_messages(message)
        full_response = ""
        for chunk in self.llm.stream(messages):
            content = getattr(chunk, "content", "")
            if isinstance(content, str):
                full_response += content
                yield content

        self.history.append(HumanMessage(content=message))
        self.history.append(AIMessage(content=full_response))

    def get_history(self) -> list[dict[str, Any]]:
        """Returneaza istoricul serializabil."""
        return [
            {"type": msg.type, "content": msg.content}
            for msg in self.history
        ]

    def clear_history(self) -> None:
        """Reset complet al conversatiei."""
        self.history = [SystemMessage(content=self.system_prompt)]

    def is_litellm_running(self) -> bool:
        """Check if LiteLLM server is running."""
        try:
            response = httpx.get(f"{BASE_URL}/health/liveliness", timeout=2.0)
            return response.status_code == 200
        except Exception:
            return False


async def main():
    agent = QAAgent(
        llm=build_llm(),
        system_prompt=SYSTEM_PROMPT,
    )
    print(f"LiteLLM available: {agent.is_litellm_running()}")
    print(f"Model: {MODEL}")
    print(f"System: {SYSTEM_PROMPT}")
    print("\nTesting chat...")
    response = agent.chat(MESSAGE)
    print(f"Response: {response}")


if __name__ == "__main__":
    asyncio.run(main())