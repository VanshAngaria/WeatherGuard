from app.llm.client import get_llm_client
from google.genai import types
import json

client = get_llm_client()
queries = [
    "zirakpur ka mausam kaisa hai",
    "weather of zirakpur",
    "is it safe to go outside of mumbai",
    "quel est le temps a Paris aujourd'hui?",
    "clima en Barcelona para correr",
    "can I visit New York tomorrow morning?",
    "Zirakpur me barish hogi kya",
    "Roorkee me ghumne ja sakte hai?",
    "Delhi ka mausam"
]

prompt_tmpl = """You are a weather-safety assistant's multilingual intent classifier.
Extract information from the user message regardless of language (English, Hindi, Hinglish, Spanish, French, etc.) or phrasing.
Respond with ONLY valid JSON:
{
  "activity_categories": [],
  "mode": null,
  "group": null,
  "location": null,
  "time_context": null,
  "is_follow_up": false,
  "raw_activity": null
}
"""

for q in queries:
    res = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents=f"{prompt_tmpl}\nUser message: {q}",
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0)
    )
    print(f"{q:45} -> {res.text.strip()}")
