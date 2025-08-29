# Services/Weather_service.py
import os
import requests
from dotenv import load_dotenv

load_dotenv()
ENV_OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_weather(location: str, api_key: str | None = None):
    """
    Fetch current weather for a given location using OpenWeather API.
    Returns a short natural language summary.
    Accepts optional api_key; falls back to .env if not provided.
    """
    key = api_key or ENV_OPENWEATHER_API_KEY
    if not key:
        return "Weather service is not configured. Missing API key."

    try:
        url = (
            f"http://api.openweathermap.org/data/2.5/weather"
            f"?q={location}&appid={key}&units=metric"
        )
        resp = requests.get(url, timeout=8)
        data = resp.json()

        if resp.status_code != 200:
            return f"Couldn't fetch weather for {location}. {data.get('message', '')}"

        weather_desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]

        return f"The weather in {location} is {weather_desc}, {temp}°C (feels like {feels_like}°C)."

    except Exception as e:
        return f"Error fetching weather: {str(e)}"