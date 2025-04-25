from pdf_processor import process_all_pdfs_in_folder
from dotenv import load_dotenv
import os
from openai import OpenAI
import json
from pathlib import Path

# Load environment just in case
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

# Folder with your PDFs
pdf_folder_path = Path(__file__).resolve().parent/"pdf"

print("Fetching PDF's to learn from: ",pdf_folder_path)

retrievers = process_all_pdfs_in_folder(pdf_folder_path)

print(f"\nTotal retrievers created: {len(retrievers)}")

print(f"\n************Starting Chat************")
print(f"\n************Type Bye To End Chat************")

client = OpenAI()


def get_chunks(question: str):
    relevant_chunks=[]
    for ret in retrievers:
        results = ret.similarity_search(query=question, k=3)
        for res in results:
            relevant_chunks.append(res.page_content.strip())
    return relevant_chunks


base_system_prompt = """

You are an AI assistant who is expert in answering questions for users. You only answer questions 
available in your context and does not answer if nothing is found in your context.

Context:
{relevant_chunks}

Rules:
- Use Available context only to solve user queries
- If Context is empty return - "I do not know this" as answer
- Provide response in JSON format

Output Format (strict JSON):
{{ "step": "string", "content": "string" }}

Example:
Input: Tell me about Framemaker?
Output: {{ step: "answer", "function":"get_chunks", "input":"Tell me about Framemaker?",content: "For a FrameMaker user, Structured FrameMaker is the easiest way to experiment with structured documents. 
It comes with ready-made templates that illustrate the extra value structure gives you. If you currently use 
unstructured FrameMaker, just open the Preferences dialog box and set the Product Interface to Structured 
FrameMaker. This won’t change any of your existing work, and you can still use all the unstructured features 
you know." }}

Example:
Input: Bye?
Output: {{ "step": "end", "content": "GoodBye!!" }}

"""

# Start the conversation loop

messages = []

while True:
    user_input = input("> ")

    if user_input.lower().strip() in ["bye", "exit", "quit"]:
        print('🧠: GoodBye!!')
        break

    relevant_chunks = get_chunks(user_input)
    context = "\n\n".join(relevant_chunks)

    if not context.strip():
        context = "I do not know this."

    system_prompt = base_system_prompt.format(relevant_chunks=context)
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": user_input})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        response_format={"type": "json_object"},
        messages=messages
    )

    parsed_response = json.loads(response.choices[0].message.content)
    messages.append({"role": "assistant", "content": json.dumps(parsed_response)})

    print(f'🧠: {parsed_response.get("content")}')

    if parsed_response.get("step") == "end":
        print("Chat ended. 👋")
        break
