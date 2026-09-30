import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=api_key)

# FIX 1: Use a production-ready model supported on Groq
model = "openai/gpt-oss-120b"

system_prompt = """
You are an AI assistant who is an expert in breaking down complex problems and resolving the user query.
Follow these steps in sequence: "think", "analyze", "output", "validate", and finally "result".

Rules:
1. Follow the strict json output structure as per output schema.
2. Always perform ONE step at a time and stop. Wait for the user or program state loop progression.
3. You must respond strictly in a valid JSON object matching the schema below. Do not output anything else.

Output Schema: {"step": "string", "content": "string"}

Example output format:
{"step": "analyze", "content": "Alright! The user is interested in a maths query."}
"""

messages = [{"role": "system", "content": system_prompt}]

# Get user input first
query = input("< ")
messages.append({"role": "user", "content": query})

while True:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        response_format={"type": "json_object"},
    )

    # FIX 2: Initialize an empty string to accumulate text tokens
    full_response_text = ""

    for chunk in response:
        # Check standard SDK syntax path for deltas
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_response_text += content
            # Visual anchor: Print text chunks in real-time as they load
            print(content, end="", flush=True)

    print()  # Add newline after streaming completes

    # FIX 3: Parse JSON safely outside the streaming loop
    try:
        parsed_response = json.loads(full_response_text)

        # Handle structural list arrays if the model wraps its object
        if isinstance(parsed_response, list):
            parsed_response = parsed_response[0] if len(parsed_response) > 0 else {}

        # Append completed message to historical state once per model invocation
        messages.append({"role": "assistant", "content": json.dumps(parsed_response)})

        step = parsed_response.get("step")
        step_content = parsed_response.get("content", "")

        # FIX 4: Corrected formatting checks using the parsed object variables
        if step != "result":
            print(f"🧠 [{step.upper()}]: {step_content}\n")
            # Automatically feed a hidden prompting agent pulse to trigger the next step sequence
            messages.append(
                {
                    "role": "user",
                    "content": "Proceed to the next logical step sequence balance.",
                }
            )
            continue
        else:
            print(f"🤖 [FINAL RESULT]: {step_content}")
            break

    except json.JSONDecodeError:
        print(
            f"⚠️ Error parsing response token structure. Raw payload was: {full_response_text}"
        )
        break
