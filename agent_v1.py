import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

load_dotenv()

@tool
def calc(expression: str) -> str:
    """Evaluate a basic math expression like '12*7 + 3'."""
    try:
        # For learning only; later replace with a safe expression parser.
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"

@tool
def summarize(text: str) -> str:
    """Summarize a piece of text into 3-5 bullet points."""
    # We’ll use the LLM for summary, but keep it as a “tool” to show tool-calling structure.
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Summarize into 3-5 bullet points:\n\n{text}"
    return llm.invoke([HumanMessage(content=prompt)]).content

def main():
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    tools = [calc, summarize]
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt="You are a helpful AI agent. Use tools when needed. If a tool is useful, call it.",
    )

    print("\nAI Agent v1 (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("exit", "quit"):
            break

        result = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
        messages = result.get("messages", [])
        reply = "No response."

        for message in reversed(messages):
            role = getattr(message, "type", None) or getattr(message, "role", None)
            if role in ("ai", "assistant"):
                content = getattr(message, "content", "")
                if isinstance(content, str):
                    reply = content
                elif isinstance(content, list):
                    text_parts = [
                        part.get("text", "")
                        for part in content
                        if isinstance(part, dict) and part.get("type") == "text"
                    ]
                    reply = "\n".join(part for part in text_parts if part).strip() or str(content)
                else:
                    reply = str(content)
                break

        print("\nAgent:", reply, "\n")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env")
    main()
