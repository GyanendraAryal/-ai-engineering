import ast
import json
import math
import operator
import ssl
import certifi
import os
import re
from typing import Any
from urllib.request import Request, urlopen

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

MODEL = "openai/gpt-oss-120b"
MAX_TOOL_ROUNDS = 8
TAVILY_SEARCH_URL = "https://api.tavily.com/search"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information and return an answer with source links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to look up on the web.",
                    }
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": (
                "Evaluate a basic arithmetic expression using addition, subtraction, "
                "multiplication (including x or ×), and division."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "An arithmetic expression, such as '2 x 2' or '(10 + 4) / 2'.",
                    }
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
]


def calculate(expression: str) -> int | float:
    """Safely evaluate a basic arithmetic expression without using eval."""
    if not expression or len(expression) > 256:
        raise ValueError("The expression must contain 1 to 256 characters.")

    normalized = expression.replace("×", "*")
    normalized = re.sub(r"(?<=[0-9.)])\s*[xX]\s*(?=[0-9.(])", "*", normalized)
    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError as error:
        raise ValueError(f"Invalid arithmetic expression: {expression!r}") from error

    if sum(1 for _ in ast.walk(parsed)) > 100:
        raise ValueError("The expression is too complex.")

    binary_operations: dict[type[ast.operator], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
    }
    unary_operations: dict[type[ast.unaryop], Any] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def evaluate(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return evaluate(node.body)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in binary_operations:
            left = evaluate(node.left)
            right = evaluate(node.right)
            result = binary_operations[type(node.op)](left, right)
            if isinstance(result, float) and not math.isfinite(result):
                raise ValueError("The result is not a finite number.")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in unary_operations:
            return unary_operations[type(node.op)](evaluate(node.operand))
        raise ValueError("Only basic arithmetic with numbers and parentheses is supported.")

    return evaluate(parsed)


def web_search(query: str) -> str:
    """Search Tavily and return a concise result set for the language model."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY is missing from the environment.")
    if not query.strip():
        raise ValueError("The web search query cannot be empty.")

    payload = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 5,
            "include_answer": True,
        }
    ).encode("utf-8")
    request = Request(
        TAVILY_SEARCH_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=20, context=ssl_context) as response:
        data = json.loads(response.read().decode("utf-8"))

    results = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "content": result.get("content", ""),
        }
        for result in data.get("results", [])
    ]
    return json.dumps(
        {"answer": data.get("answer"), "results": results},
        ensure_ascii=False,
    )


def run_agent(query: str) -> str:
    """Send a query to Groq and automatically execute any requested tools."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is missing from the environment.")
    if not query.strip():
        raise ValueError("The query cannot be empty.")

    client = Groq(api_key=api_key)
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "Answer the user's request accurately. Use web_search for current or "
                "factual information that needs web research, and cite source URLs from "
                "its results. Use calculate for arithmetic instead of doing calculations "
                "mentally. Choose tools automatically when they are useful."
            ),
        },
        {"role": "user", "content": query},
    ]
    tool_functions = {"web_search": web_search, "calculate": calculate}

    for _ in range(MAX_TOOL_ROUNDS):
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        message = completion.choices[0].message
        tool_calls = message.tool_calls

        if not tool_calls:
            if message.content is None:
                raise RuntimeError("Groq returned neither an answer nor a tool call.")
            return message.content

        messages.append(
            {
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                    for tool_call in tool_calls
                ],
            }
        )

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function = tool_functions.get(function_name)
            if function is None:
                raise RuntimeError(f"Groq requested an unknown tool: {function_name}")

            arguments = json.loads(tool_call.function.arguments)
            if not isinstance(arguments, dict):
                raise ValueError(f"Arguments for {function_name} must be a JSON object.")
            result = function(**arguments)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result if isinstance(result, str) else str(result),
                }
            )

    raise RuntimeError(f"Groq exceeded the limit of {MAX_TOOL_ROUNDS} tool rounds.")


def main() -> None:
    query = input("Ask me something: ").strip()
    print(run_agent(query))


if __name__ == "__main__":
    main()
