## Tracking Collections - http://localhost:6333/dashboard#/collections
## Ollama: http://localhost:11434
## Qdrant UI (if you enabled): http://localhost:6333


from indexing.pdf_processor import process_all_pdfs_in_folder
from dotenv import load_dotenv
import os,re
from openai import OpenAI
import json
from pathlib import Path
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from retreival.fanout_rrf_query import fanout_rrf_query_multiqueries
from retreival.subquery_fanout_llm import generate_query_variants
from mem0 import MemoryClient


# Load environment just in case
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

OLLAMA_HOST = os.getenv("OLLAMA_URL")
if not OLLAMA_HOST:
    raise ValueError("Missing OLLAMA_HOST in .env file")

MEM0_KEY = os.getenv("MEM0_API_KEY")
if not OLLAMA_HOST:
    raise ValueError("Missing MEM0_KEY in .env file")


# Initialize once
embedder = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)
DIMENSIONS = 768
# embedder = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     api_key=os.getenv("OPENAI_API_KEY")
# )
#DIMENSIONS = 1536

# Folder with your PDFs
pdf_folder_path = Path(__file__).resolve().parent.parent /"pdf"

retrievers = process_all_pdfs_in_folder(pdf_folder_path,True,False,DIMENSIONS,embedder)

print(f"\nTotal retrievers created: {len(retrievers)}")

print(f"\n************Starting Chat************")
print(f"\n************Type Bye To End Chat************")

#client = OpenAI()
client = ChatOllama(
    base_url=OLLAMA_HOST,  # Ollama via Docker
    model="llama3",                     # Or another pulled model (like mistral, codellama, etc.)
    temperature=0
)

mem0_client = MemoryClient(api_key=MEM0_KEY)

def is_json(myjson: str) -> bool:
    try:
        json.loads(myjson)
        return True
    except ValueError:
        return False


def get_chunks(question: str):
    queries = generate_query_variants(question)
    # print("\nGenerated Query Variants:")
    # for idx, q in enumerate(queries, 1):
    #     print(f"  Q{idx}: {q}")
    relevant_chunks= fanout_rrf_query_multiqueries(queries,retrievers,20,10)
    print(f"Relevant Chunks returned: {len(relevant_chunks)}")
    # print("\nChunks:")
    # print(relevant_chunks)
    # Code for basic rag without any optimization like Fanout, Hyde
    # for ret in retrievers:
    #     results = ret.similarity_search(query=question, k=3)
    #     for res in results:
    #         relevant_chunks.append(res.page_content.strip())
    return relevant_chunks

base_system_prompt = """

You are an AI assistant who is expert in answering questions for users. 
Use as much context available with you.
and prefer long, detailed answers if relevant context supports it. Summarize across multiple chunks if needed.
Use examples if available in context to explain questions. If examples are not available in context skip examples.
You only answer questions 
available in your context and do not answer if nothing is found in your context. 

Context:
{relevant_chunks}

Rules:
- Use available context only to solve user queries.
- If context is empty return - "I do not know this" as answer.
- Combine and synthesize information across multiple context parts if needed.
- Provide response in strict JSON format.

IMPORTANT: You must only output a single JSON object. No additional text, no explanation, no markdown.
If you do not know the answer, respond with {{"step": "answer", "content": "I do not know"}} exactly.

IMPORTANT: You must return a valid JSON object with the format:
{{ "step": "answer", "content": "..." }}
Do NOT return plain text or natural language. If unsure, say:
{{ "step": "answer", "content": "I do not know" }}

Output Format (strict JSON):
{{ "step": "string", "content": "string" }}

Example:
Input: Tell me about Framemaker?
Output: {{ "step": "answer", "function":"get_chunks", "input":"Tell me about Framemaker?", "content": "For a FrameMaker user, Structured FrameMaker is the easiest way to experiment with structured documents. It comes with ready-made templates that illustrate the extra value structure gives you. If you currently use unstructured FrameMaker, just open the Preferences dialog box and set the Product Interface to Structured FrameMaker. This won’t change any of your existing work, and you can still use all the unstructured features you know." }}

Example:
Input: Bye
Output: {{ "step": "end", "content": "GoodBye!!" }}

Example:
Input: Hi
Output: {{ "step": "answer", "content": "How can I help you today?" }}

Input: Who was Hitler?
Output: {{ "step": "answer", "function":"get_chunks", "input":"Who was Hitler?", "content": "I don’t know" }}

"""

# Start the conversation loop

messages = []

while True:
    user_input = input(">> ")

    if user_input.lower().strip() in ["bye", "exit", "quit"]:
        print('🧠: GoodBye!!')
        break

    mem0_context = mem0_client.search(user_input, user_id="kapil")
    mem0_context_texts = [mem.get("content", "") for mem in mem0_context if isinstance(mem, dict)]
    relevant_chunks = get_chunks(user_input)
    all_texts = mem0_context_texts + relevant_chunks
    context = "\n\n".join(set(all_texts))


    if not context.strip():
        context = "I do not know this."

    system_prompt = base_system_prompt.format(relevant_chunks=context)
    
    #For Gemini/Llama3
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_input)
    ]

    print(f"messages:{messages}")

    try:
        response = client.invoke(messages)
        json_str = re.sub(r"^```(?:json)?\n|\n```$", "", response.content.strip())
        if is_json(json_str):
            parsed_response = json.loads(json_str)
        else:
            # fallback: wrap plain text into your json format
            parsed_response = {
            "step": "answer",
            "content": response.content.strip()
            }
        
    except Exception as e:
        print("⚠️ Error parsing response. Raw response:")
        print (e)
        print(response)
        continue
    # For Open AI
    # messages = [{"role": "system", "content": system_prompt}]
    # messages.append({"role": "user", "content": user_input})
    # response = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     response_format={"type": "json_object"},
    #     messages=messages
    # )
    # parsed_response = json.loads(response.choices[0].message.content)
    # messages.append({"role": "assistant", "content": json.dumps(parsed_response)})
    
    mem0_client.add(
    [
        {
            "role": "system" if isinstance(msg, SystemMessage)
                     else "user" if isinstance(msg, HumanMessage)
                     else "assistant",
            "content": msg.content.strip()
        }
        for msg in messages if hasattr(msg, "content") and msg.content.strip()
    ],
    user_id="kapil",
    output_format="v1.1"
    )

    print(f'🧠: {parsed_response.get("content")}')

    if parsed_response.get("step") == "end":
        print("Chat ended. 👋")
        break
