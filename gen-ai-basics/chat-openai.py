from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI()


result = client.chat.completions.create(

    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content":"Hey User"}
    ]
)
print("Hello")
print(result.choices[0].message.content)