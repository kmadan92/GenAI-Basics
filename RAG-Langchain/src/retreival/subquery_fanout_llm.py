from langchain_google_genai import ChatGoogleGenerativeAI
import os, json
from pathlib import Path
from dotenv import load_dotenv

# Load environment just in case
env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
print(env_path)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Missing GOOGLE_API_KEY in .env file")


client = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    temperature=0.3,
    google_api_key=GOOGLE_API_KEY
)

def generate_query_variants(query: str) -> list:
   
    prompt = f"""
    You are an AI Assistant that helps in creating variants of query that user has asked. You reformulate the use query into 3 alternate variants having same intent but different wording.
    Return them strictly as a JSON array of strings ONLY, nothing else.

    Original Query: "{query}"
    """
    try:
        response = client.invoke([{"role": "user", "content": prompt}])
        output = response.content.strip()
        #print(output)
        if output.startswith("```"):
            output = "\n".join(output.split("\n")[1:-1]).strip()
        
        variants = json.loads(output)
        return [query] + variants[:3]  # Return original + 3 variants
    except Exception as e:
        print("⚠️ Failed to reformulate query:", e)
        return [query]
