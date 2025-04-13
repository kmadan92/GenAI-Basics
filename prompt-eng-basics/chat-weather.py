import json

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

def get_weather(city: str):
    return "Its 30 degree celcius and hot in the day" # API call will go here

available_tools = {
    "get_weather":{
        "fn":get_weather,
        "description" : "This tool taken city as input and gives weather data of that city as output"
    }
}

system_prompt = """
You are an AI agent who is specialized in resolving user queries. You analyze user queries break it into smaller steps and 
return answers to user queries.

Rules:
- Follow step by step approach to solve user queries
- Use Available tools if required to solve user queries
- Provide response in JSON format

Output Format:
{{ step: "string", content: "string" }}

Example:
Inout: What is the weather in Bengaluru
Output: {{"step":"analyze","content": "I need to get weather data"}}
Output: {{"step":"analyze","content": "I need to get weather data of Bengaluru city"}}
Output: {{"step":"action", "function":"get_weather", "input":"Bengaluru","content":"Thinking........"}}
Output: {{"step":"observe", "content": "I will use the available tools to get weather data now"}}
Output: {{"step":"output", "content": "The weather of Bengaluru in 28 degree celcius and its hot in afternoon"}}
"""

messages = [
    { "role": "system", "content": system_prompt },
]


query = input("> ")
messages.append({ "role": "user", "content": query })


while True:
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=messages
    )

    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({ "role": "assistant", "content": json.dumps(parsed_response) })

    if parsed_response.get("step") != "output":
        print(f"🧠: {parsed_response.get("content")}")
        continue
    
    if parsed_response.get("step") == "action":
        tool_function = parsed_response.get("function")
        tool_input = parsed_response.get("input")
        
        if(available_tools.get(tool_function,False)!=False):
            output = available_tools.get(tool_function).get("fn")(tool_input)
            messages = messages.append({"role":"assistant","content":output})

    if parsed_response.get("step") == "output":
        print(f"🧠: {parsed_response.get("content")}")
        break


