from datetime import datetime

import pytz
import json



# Note: function names must same as the function name in the json file and need unique id(id definition at json file) at last of the function name.
# Example : get_current_weather_42c9af => get_current_weather + _ + 42c9af

def get_current_weather_42c9af(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "paris" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "41摄氏度，全天晴天，紫外线指数最高10级，阵风风速最高3公里/时"}, ensure_ascii=False)

def get_current_weather_25b737(location, unit):
    """Get the current weather in a given location"""
    if "paris" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "41摄氏度，全天晴天，紫外线指数最高10级，阵风风速最高3公里/时"}, ensure_ascii=False)

def get_current_datetime_f77b31():
    """Get the current datetime"""
    return json.dumps({"current_datetime": datetime.now(pytz.timezone('Asia/Shanghai')).strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False)

