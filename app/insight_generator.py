import os
import json
import ollama
from dotenv import load_dotenv

load_dotenv()

INSIGHT_MODEL = os.getenv("OLLAMA_INSIGHT_MODEL", "gemma3:4b")

def generate_insight(question: str, sql: str, results: list) -> str:
    
    sample = results[:20]

    prompt = f"""You are a professional business data analyst.

A user asked: "{question}"

The following SQL was run:
{sql}

The results returned:
{json.dumps(sample, indent=2, default=str)}

Write a clear, concise business insight in 3-5 sentences.
- Highlight the most important finding
- Use plain English, no technical jargon
- If numbers are involved, mention the key figures
- End with one actionable recommendation
"""

    response = ollama.chat(
        model=INSIGHT_MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"].strip()