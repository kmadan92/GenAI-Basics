from openai import OpenAI
from dotenv import load_dotenv
import os

from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent/".env")

client = OpenAI()

system_prompt = "You are an AI assistant specialized in maths domain. DO not give any answer outside Maths domain" \
"" \
"" \
"Example: 2+2=4 is done by adding 2 and 2" \
"" \
"Example: 10*5=50 is done by multiplying 5 with 10"


result = client.chat.completions.create(

    model="gpt-3.5-turbo",
    max_tokens=100, # As API's are charged for tokens. We can provide max tokens to generate in outputs
    messages=[
        {"role": "system", "content":system_prompt}, #Few shot prompting which uses examples as given (Chain of thought)
        {"role": "user", "content":"Hey User"}  # zero shot prompting if no system prompt is given
    ]
)

print(result.choices[0].message.content)


# Other types of prompting
# self-consistent prompting: Gather answers from different sources and then provide best answer
# persona based prompting: giving a persona to chatbot so that it talks like it
# role play promting: giving a chatbot a role like a teacher etc and interacts accordingly
# Combining all types of prompting, a good chatbot can be developed