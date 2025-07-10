from langchain_ollama import ChatOllama
import os, json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

# Load environment just in case
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")

OLLAMA_HOST = os.getenv("OLLAMA_URL")
if not OLLAMA_HOST:
    raise ValueError("Missing OLLAMA_HOST in .env file")


client = ChatOllama(
    base_url=OLLAMA_HOST,  # Ollama via Docker
    model="llama3",        # Or another pulled model (like mistral, codellama, etc.)
    temperature=0
)

def generate_query_variants(query: str) -> list:
   
    prompt = f"""
    You are an AI Assistant that helps in creating variants of query that user has asked. You reformulate the use query into 3 alternate variants having same intent but different wording.
    Return them strictly as a JSON array of 3 strings with no extra text, no explanations, no trailing characters..

    Original Query: "{query}"

    Example 1:
Original Query: "What is machine learning?"
Output:
[
  "Can you explain what machine learning is?",
  "Tell me about machine learning.",
  "What does machine learning mean?"
]

Example 2:
Original Query: "How do I bake a chocolate cake?"
Output:
[
  "What are the steps to bake a chocolate cake?",
  "Can you tell me how to make a chocolate cake?",
  "How do I prepare a chocolate cake?"
]

    """
    try:
        response = client.invoke([{"role": "user", "content": prompt}])
        output = response.content.strip()
        # print(output)
        if output.startswith("```"):
            output = "\n".join(output.split("\n")[1:-1]).strip()
        
        variants = json.loads(output)
        return [query] + variants[:3]  # Return original + 3 variants
    except Exception as e:
        print("⚠️ Failed to reformulate query:", e)
        return [query]
