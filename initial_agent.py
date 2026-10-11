# QA agent / LLM orchestrator with tool execution and prompt management.
from __future__ import annotations
import asyncio
import json
import os
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from prompts import PromptRegistry
from tools.tool_wrapper import ToolWrapper


BASE_URL = "http://localhost:4000"
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
INITIAL_SYSTEM_PROMPT = (
    "Esti un agent de extracție specializat în extragerea datelor structurate din documente legale, cum ar fi contracte și facturi, într-un format json."
)


def build_llm() -> BaseChatModel:
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")

    return ChatGoogleGenerativeAI(
        model=MODEL,
        google_api_key=GEMINI_API_KEY,
        temperature=0.7,
        max_output_tokens=1024,
    ).bind_tools(ToolWrapper.catalog())

class QAAgent:
    """ Un agent conversațional cu istoric - un wrapper peste un ChatModel. """

    def __init__(self, llm: BaseChatModel, system_prompt: str, max_iterations: int = 3):
        """Initialize the QAAgent with a chat model, system prompt, and max iterations."""
        self.llm: BaseChatModel = llm
        self.system_prompt: str = system_prompt
        self.max_iterations = max_iterations
        self.prompt_registry = PromptRegistry("prompts")
        self.history: list[BaseMessage] = [SystemMessage(content=system_prompt)]

    async def chat(self, message: str) -> str:
        """Send a message, receive a complete response, update history."""
        messages = self._build_messages(message)
        answer = await self.react_loop(self.llm, messages)
        self.history = messages
        self.history.append(AIMessage(content=answer))
        return answer
    
    async def react_loop(self, llm, mesaje: list) -> str:
        """ Loop Think → Act → Observe. """
        for _ in range(self.max_iterations):
            raspuns = await llm.ainvoke(mesaje)
            mesaje.append(raspuns)

            tool_calls = getattr(raspuns, "tool_calls", None) or []

            if not tool_calls:
                return getattr(raspuns, "content", "")

            rezultate = await self.execute_all_tools(tool_calls)
            for mesaj in rezultate:
                mesaje.append(ToolMessage(
                    content=mesaj["content"],
                    tool_call_id=mesaj["tool_call_id"],
                    name=mesaj.get("name"),
                ))

        raise RuntimeError("Max iterations atins fără răspuns final")

    async def execute_all_tools(self, tool_calls: list) -> list:
        """Execute all tool calls concurrently and return their results."""
        tasks = [self.execute_tool_async(tc) for tc in tool_calls]
        return await asyncio.gather(*tasks)
    
    @staticmethod
    def _normalize_tool_call(tool_call: dict) -> tuple[str, dict, str | None]:
        """Normalize tool call payloads across different provider formats."""
        if not isinstance(tool_call, dict):
            raise ValueError(f"Invalid tool call payload: {tool_call!r}")

        name = tool_call.get("name") or tool_call.get("function", {}).get("name")
        if not name:
            raise ValueError(f"Tool call without name: {tool_call!r}")

        raw_args = tool_call.get("args")
        if raw_args is None:
            raw_args = tool_call.get("arguments", {})

        if isinstance(raw_args, str):
            try:
                parsed_args = json.loads(raw_args)
            except json.JSONDecodeError:
                parsed_args = {"value": raw_args}
            raw_args = parsed_args

        if not isinstance(raw_args, dict):
            raw_args = {"value": raw_args}

        call_id = tool_call.get("id") or tool_call.get("call_id")
        return name, raw_args, call_id

    async def execute_tool_async(self, tool_call: dict) -> dict:
        """Execute a single tool call asynchronously and return the result."""
        try:
            name, args, call_id = self._normalize_tool_call(tool_call)
        except ValueError as exc:
            return {"tool_call_id": None, "name": "unknown", "content": str(exc)}

        try:
            rezultat = await asyncio.to_thread(ToolWrapper.call, name, args)
            return {"tool_call_id": call_id, "name": name, "content": str(rezultat)}
        except Exception as exc:
            return {
                "tool_call_id": call_id,
                "name": name,
                "content": f"Eroare execuție '{name}': {exc}",
            }

    def _build_messages(self, message: str) -> list[BaseMessage]:
        """Build the message list for the LLM, including system prompt and history."""
        selected_system_prompt = self.build_system_prompt(message)
        self.system_prompt = selected_system_prompt
        non_system_history = [msg for msg in self.history if not isinstance(msg, SystemMessage)]
        self.history = [SystemMessage(content=selected_system_prompt)] + non_system_history
        return self.history + [HumanMessage(content=message)]
        
    def build_system_prompt(self, message: str, role: str = "lawyer", domain: str = "contracts") -> str:
        """Render a task-specific system prompt from the prompt YAML files."""
        prompt_name = self.prompt_registry.select_prompt_name(message)
        render_kwargs = {"role": role, "domain": domain}

        if prompt_name in {"ask_planner", "ask_summary"}:
            render_kwargs["max_cuvinte"] = 80
        elif prompt_name == "ask_analyst":
            render_kwargs["max_cuvinte"] = 120

        return self.prompt_registry.render(prompt_name, **render_kwargs)

async def main():
    agent = QAAgent(llm=build_llm(), system_prompt=INITIAL_SYSTEM_PROMPT)

    reply = await agent.chat("Foloseste search documents documente pentru facturi si extrage datele contractului de imobil?")
    print(f"Agent: {reply}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.close()