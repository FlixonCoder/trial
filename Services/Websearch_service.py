# Services/Websearch_service.py
import os
from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv()
ENV_TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def web_search(query: str, api_key: str | None = None) -> dict:
    """
    Perform a Tavily web search.
    Returns a simplified dictionary with summarized results.
    Accepts optional api_key; falls back to .env if not provided.
    """
    print(f"[Tool Call] web_search({query})")
    try:
        key = api_key or ENV_TAVILY_API_KEY
        if not key:
            return {"error": "Websearch service not configured. Missing API key."}

        client = TavilyClient(api_key=key)
        response = client.search(query)
        results = []

        for item in response.get("results", []):
            results.append({
                "title": item.get("title"),
                "url": item.get("url"),
                "content": item.get("content"),
            })

        print(f"[Tool Response] {results[:2]} ...")  # log preview
        return {"results": results}

    except Exception as e:
        print(f"[WebSearch error] {e}")
        return {"error": str(e)}