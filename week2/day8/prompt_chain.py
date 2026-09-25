import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq
from time import sleep

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("API key kaha hai laude")

client = Groq(api_key=api_key)
model = "openai/gpt-oss-120b"


JD = """
We are hiring a Backend Python Developer.

Requirements:
- Strong Python
- FastAPI or Django
- PostgreSQL
- Docker
- AWS
- REST APIs
- 2+ years of experience
"""
RESUME = """
Name: Rahul Sharma

Experience:
3 years as a Software Developer.

Skills:
Python, FastAPI, MySQL, Docker,
REST APIs, Git

Projects:
Built a food delivery backend using
FastAPI and MySQL.

Deployed applications using Docker.
"""


def ask_llm(system_prompt, user_prompt):
    sys_msg = {
        "role": "system",
        "content": system_prompt,
    }
    user_msg = {
        "role": "user",
        "content": user_prompt,
    }
    messages = [sys_msg, user_msg]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    answer = response.choices[0].message.content
    return answer


def res_extract(RESUME):
    # extract skills from resume
    system_prompt = """
    You are a professional HR assistant. Extract the skills from the candidates resume provided.
    Only return the skills no other information. Do not invent any skillsby yourself.
    Output Format:
    Skills should be separated by commas. Just return comma separated skills do not return any other filler information
    """
    user_prompt = f"""
    Extract the skills from this resume
    {RESUME}
    """
    return ask_llm(system_prompt, user_prompt)


def jd_extract(JD):
    print("step2")
    system_prompt = """
    You are a professional HR assistant. Extract the skills from the Job description  provided.
    Only return the skills no other information. Do not invent any skills by yourself.
    Output Format:
    Skills should be separated by commas. Just return comma separated skills do not return any other filler information
    """
    user_prompt = f"""
    Extract the skills from this JD
    {JD}
    """
    return ask_llm(system_prompt, user_prompt)


def match(candidate, JD):
    system_prompt = """
    Your are professional HR assistant. compare the skills of candidate and the skills required in the JD and produce a final score between 1 and 100 also produce a short verdict whether a candidate is good fit for the role
    """
    user_prompt = f"""
    Compare and match the skills.
    JD:{JD}
    Candidate:{candidate}
    """
    return ask_llm(system_prompt, user_prompt)


candidate = res_extract(RESUME)
sleep(3)
jd = jd_extract(JD)
sleep(3)
score = match(candidate, jd)
sleep(3)
print(score)
