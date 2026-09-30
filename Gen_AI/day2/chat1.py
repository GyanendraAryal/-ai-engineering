import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key kaha hai bhai")

client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"

system_prompt = """
You are an AI assistant who is expert in breaking down complex problems and then resolve the user query.
For the given user input analyze the input and breakdown the problem step by step.
Atleast think in 5-6 steps on how to solve the problem before solving it down.
The steps are you get a user input, you analyze, you think, you think again     for several times and then return an output with explaination. And then finally you validate hte output as well before giving final results.
You must respond strictly in valid JSON format.


Follow these seps in sequence that is "think", "analyze","output","validate" and finally "result",

Rules:
1. Follow the strict json output as per output schema.
2. Always perform ne step at a time and wait for the next input.
3. Carefully analyze the user query.

Output: {{step:"string","content":"string"}}

Example:
Input: what is 2 + 2.
Output:{{ step:"analyse", content:"Alright! The user is interested in maths query and he is asking a basic arthmetic operation."}}
Output:{{ step:"think", content:"To perform an addition i must go from left to right and add all the operands."}}
Output:{{ step:"output", content:"4"}}
Output:{{ step:"validate", content:"Seems like 4 is correct answer for 2 + 2"}}
Output:{{ step:"result", content:"2 + 2 and 4 that is calculated by adding all numbers"}}

Input: Why is sky blue?
Output: Bruh? Are you alright? Is is a math query?
"""
system_message = {
    "role": "system",
    "content": system_prompt,
}
user_prompt = """
What is 3 + 4 * 5?
"""
user_message = {
    "role": "user",
    "content": user_prompt,
}

messages = [
    system_message,
    user_message,
    #
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "step": "analyse",
                "content": "User asks for the result of the arithmetic expression 3 + 4 * 5, requiring evaluation with proper operator precedence.",
            }
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "step": "think",
                "content": "Apply operator precedence: multiplication before addition. Compute 4 * 5 = 20, then add 3, resulting in 23.",
            }
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({"step": "output", "content": "23"}),
    },
]
response = client.chat.completions.create(
    model=model,
    messages=messages,
    stream=True,
    response_format={
        "type": "json_object",
    },
)
# answer = response.choices[0].message.content
for chunk in response:
    content = chunk.choices[0].delta.content
    if content:
        print(content, end="", flush=True)
print("\n\nStream complete")
