# QA agent / LLM orchestrator with tool execution and prompt management.
from __future__ import annotations
import asyncio
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from prompts import PromptRegistry
from tools.tool_wrapper import ToolWrapper

BASE_URL = "http://localhost:4000"
MODEL = "llama3.2"
INITIAL_SYSTEM_PROMPT = (
    "You are a helpful assistant for a law firm. Use tools when needed."
)

def build_llm() -> BaseChatModel:
    return ChatOllama(
        model=MODEL,
        temperature=0.7,
        max_tokens=1024,
    ).bind_tools(ToolWrapper.catalog())

class QAAgent:
    """ A conversational agent with history - a wrapper over a ChatModel. """

    def __init__(self, llm: BaseChatModel, system_prompt: str, max_iterations: int = 3):
        """Initialize the QAAgent with a chat model, system prompt, and max iterations."""
        self.llm: BaseChatModel = llm
        self.system_prompt: str = system_prompt
        self.max_iterations = max_iterations
        self.prompt_registry = PromptRegistry("prompts")
        self.history: list[BaseMessage] = [SystemMessage(content=system_prompt)]
        self.is_litellm_running()  # Check if LiteLLM server is running

    async def chat(self, message: str) -> str:
        """Send a message, receive a complete response, update history."""
        messages = self._build_messages(message)
        answer = await self.react_loop(self.llm, messages)
        self.history = messages
        self.history.append(AIMessage(content=answer))
        return answer
    
    async def react_loop(self, llm, mesaje: list, max_iterations: int = 5) -> str:
        """ Loop Think → Act → Observe. """
        for _ in range(max_iterations):
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
    
    async def execute_tool_async(self, tool_call: dict) -> dict:
        """Execute a single tool call asynchronously and return the result."""
        name = tool_call.get("name") or tool_call.get("function", {}).get("name")
        args = tool_call.get("args") or tool_call.get("arguments") or {}
        call_id = tool_call.get("id") or tool_call.get("call_id")

        if not name:
            raise ValueError(f"Tool call without name: {tool_call!r}")

        rezultat = await asyncio.to_thread(ToolWrapper.call, name, args)
        return {"tool_call_id": call_id, "name": name, "content": str(rezultat)}

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

    def is_litellm_running(self) -> None:
        """Check if LiteLLM server is running."""
        try:
            response = httpx.get(f"{BASE_URL}/health/liveliness", timeout=2.0)
            if response.status_code == 200:
                return
        except Exception:
            raise RuntimeError("LiteLLM server is not running")

async def main():
    agent = QAAgent(llm=build_llm(), system_prompt=INITIAL_SYSTEM_PROMPT)

    print("Chat started. Type 'exit' to quit.")

    while True:
        user_msg = input("You: ").strip()
        if user_msg.lower() in {"exit", "quit"}:
            break

        reply = await agent.chat(user_msg)
        print(f"Agent: {reply}")


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
    loop.close()