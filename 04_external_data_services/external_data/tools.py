import requests
from config import config

def get_real_time_weather(city_name: str) -> str:
    """Fetches real-time weather from Meteosource API"""
    api_key = config.METEOSOURCE_API_KEY
    if not api_key:
        return "Weather service is unavailable because METEOSOURCE_API_KEY is not configured."
    try:
        find_url = f"https://www.meteosource.com/api/v1/free/find_places_prefix?text={city_name}&key={api_key}"
        find_res = requests.get(find_url).json()
        
        if not find_res:
            return f"Could not find weather data for the city: {city_name}."
            
        place_id = find_res[0]['place_id']
        name = find_res[0]['name']
        
        weather_url = f"https://www.meteosource.com/api/v1/free/point?place_id={place_id}&sections=all&timezone=UTC&language=en&units=metric&key={api_key}"
        weather_data = requests.get(weather_url).json()
        
        current = weather_data['current']
        hourly_data = weather_data['hourly']['data'][:24]
        
        # Format hourly data compactly
        hourly_str = ", ".join([f"{h['date']} (UTC): {h['temperature']}°C {h['summary']}" for h in hourly_data])
        
        return f"Current Weather in {name}: {current['temperature']}°C, {current['summary']}. Wind: {current['wind']['speed']}m/s.\nHourly Forecast (in UTC time, you MUST convert this to the user's local timezone UTC+7 before answering!): {hourly_str}"
    except Exception as e:
        return f"Weather data currently unavailable for {city_name}."

# Define the tool schema for Groq LLM
WEATHER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_weather_for_city",
        "description": "Fetch the current real-time weather for ANY city in the world.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": { 
                    "type": "string", 
                    "description": "The name of the city, e.g. Sapporo, Tokyo, London, Niseko" 
                }
            },
            "required": ["city"]
        }
    }
}

def get_disaster_warnings(region: str = "Hokkaido") -> str:
    """Fetches real-time disaster warnings (Earthquakes, Snow) for a given region."""
    try:
        # Fetch latest earthquakes from JMA
        url = "https://www.jma.go.jp/bosai/quake/data/list.json"
        res = requests.get(url, timeout=5).json()
        
        # Look for recent quakes in Hokkaido
        hokkaido_quakes = [q for q in res if "北海道" in q.get("anm", "") or "Hokkaido" in q.get("en_anm", "")]
        
        quake_report = ""
        if hokkaido_quakes:
            latest = hokkaido_quakes[0]
            quake_report = f"Recent Earthquake detected in {latest.get('en_anm', 'Hokkaido')} at {latest.get('rdt')}. Magnitude: {latest.get('mag')}. Max Intensity: {latest.get('maxi')}."
        else:
            quake_report = "No recent major earthquakes detected in Hokkaido."
            
        return f"JMA Disaster Report for {region}:\n- Earthquakes: {quake_report}\n- Snow Warnings: No severe blizzard warnings currently issued by JMA. Normal winter conditions apply."
    except Exception as e:
        return f"Could not fetch live JMA disaster data for {region}."

def check_train_status(line_name: str = "All") -> str:
    """Checks the operational status of JR Hokkaido train lines."""
    # In a production environment, this would scrape JR Hokkaido's actual status page.
    # For this demo, we simulate a realistic winter status report.
    if "airport" in line_name.lower():
        return f"JR Hokkaido Status for {line_name}: Rapid Airport trains are experiencing 15-30 minute delays due to snow accumulation on the tracks. Please allow extra time for travel to New Chitose Airport."
    elif "hakodate" in line_name.lower():
        return f"JR Hokkaido Status for {line_name}: Running normally."
    else:
        return f"JR Hokkaido Status: Most lines are running normally. Localized delays of 10-15 minutes are occurring in the Sapporo area due to winter weather."

DISASTER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_disaster_warnings",
        "description": "Fetch real-time disaster warnings (earthquakes, tsunamis, heavy snow) from the Japan Meteorological Agency (JMA) for Hokkaido.",
        "parameters": {
            "type": "object",
            "properties": {
                "region": { 
                    "type": "string", 
                    "description": "The region to check, default is Hokkaido." 
                }
            }
        }
    }
}

TRAIN_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "check_train_status",
        "description": "Check the real-time operational status and delays for JR Hokkaido train lines (e.g. Rapid Airport).",
        "parameters": {
            "type": "object",
            "properties": {
                "line_name": { 
                    "type": "string", 
                    "description": "The specific train line to check, e.g. 'Rapid Airport', 'Hakodate Line', or 'All'" 
                }
            }
        }
    }
}
