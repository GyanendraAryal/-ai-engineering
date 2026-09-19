import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("API key kaha hai bhai")
client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-20b"
role = "user"
prompt1 = "Hi"
prompt2 = "Explain time travel in detail but under 100 words"
prompt3 = "Write 1000 word essay on machine learning."
prompts = [prompt1, prompt2, prompt3]
for prompt in prompts:
    message = {
        "role": role,
        "content": prompt,
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model, messages=messages, max_tokens=50
    )
    usage = response.usage
    print(
        f"Prompt: {prompt}--> Your token {usage.prompt_tokens} Completion_token {usage.completion_tokens} Total tokens {usage.prompt_tokens + usage.completion_tokens} Finish Reason: {response.choices[0].finish_reason}"
    )

# message = {"role": role, "content": content}
# # Temperatue ramge is [0,2]
# response = client.chat.completions.create(model=model, messages=messages, temperature=2)
# # print(response)
# answer = response.choices[0].message.content
# print(answer)
