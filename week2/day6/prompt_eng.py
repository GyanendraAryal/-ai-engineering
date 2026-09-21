import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("API kaha hai bhai")


client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"
role = "user"


def llm_ans(prompt):
    message = {
        "role": role,
        "content": prompt,
    }
    messages = [message]
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=2
    )
    answer = response.choices[0].message.content
    return answer


# Bad Prompt
bad_prompt = """
#Role:
You are a support assistant at mobile/laptop company

#Task:
You have to classify the issue in category

#Constraints:
You have to classify the issue in one of the three category namely BILLING, TECHNICAL, RETURN.

# Output Format:
Your answer should be in one word only and that one word should be one of the category provided in Constraints.

# Example:
For instacne if a user complains saying he wants a refund then the category is return.

# Fall Back:
If the issue is not related to any of the category mentioned in the constraints then mark it as OTHERS.

This is a user complaint:
I had to pay more than the cost of the mobile phone.
"""
# I had to pay more than the cost of the mobile phone.

# if issue does not fall into mentioned category mark it as others.

# My laptop is not working
# My girlfriend left me 😔.
# I'm not happy with my laptop.

print(llm_ans(bad_prompt))
print("=================")
