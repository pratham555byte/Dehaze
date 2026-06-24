import json
from typing import Any, Dict, List

import requests
from geopy.distance import geodesic

# ---------------------------------------------------
# LOAD BLACKSPOTS
# ---------------------------------------------------

with open("config/blackspots.json") as f:

    BLACKSPOTS = json.load(f)

# ---------------------------------------------------
# GET ROAD CONTEXT
# ---------------------------------------------------

def get_road_context(lat: float, lon: float) -> Dict[str, Any]:

    current_location = (lat, lon)
    road = "Unknown"
    raw_location: Dict[str, Any] = {}

    # ---------------------------------------------------
    # REVERSE GEOCODING
    # ---------------------------------------------------

    url = (
        f"https://nominatim.openstreetmap.org/"
        f"reverse?format=jsonv2&lat={lat}&lon={lon}"
    )

    headers = {
        "User-Agent": "road-awareness-system"
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        raw_location = data
        address = data.get("address", {})
        road = address.get("road", road)
    except requests.RequestException:
        data = {}

    # ---------------------------------------------------
    # ROAD TYPE
    # ---------------------------------------------------

    if (
        "highway" in road.lower()
        or "nh" in road.lower()
    ):

        road_type = "Highway"

    else:

        road_type = "Urban Road"

    # ---------------------------------------------------
    # BLACKSPOT CHECKING
    # ---------------------------------------------------

    nearby_blackspots: List[Dict[str, Any]] = []
    hazard_types = set()

    for spot in BLACKSPOTS:
        spot_location = (
            spot["latitude"],
            spot["longitude"]
        )

        distance = geodesic(
            current_location,
            spot_location
        ).km

        if distance < 2.0:
            nearby_blackspots.append({
                "name": spot["name"],
                "distance": round(distance, 2),
                "severity": spot.get("severity", "unknown"),
                "fog_prone": bool(spot.get("fog_prone", False)),
                "rain_prone": bool(spot.get("rain_prone", False)),
                "landslide_prone": bool(spot.get("landslide_prone", False)),
                "weather_risk": spot.get("weather_risk", "unknown"),
            })

            if spot.get("fog_prone"):
                hazard_types.add("fog_prone")
            if spot.get("rain_prone"):
                hazard_types.add("rain_prone")
            if spot.get("landslide_prone"):
                hazard_types.add("landslide_prone")
            if spot.get("accident_prone"):
                hazard_types.add("accident_prone")
            if spot.get("weather_risk"):
                hazard_types.add(str(spot["weather_risk"]).lower())

    if "curve" in road_type.lower():
        hazard_types.add("curve")
    if "expressway" in road_type.lower() or "highway" in road_type.lower():
        hazard_types.add("high_speed")
    if not hazard_types:
        hazard_types.add("standard")

    # ---------------------------------------------------
    # RETURN CONTEXT
    # ---------------------------------------------------

    return {

        "latitude": lat,

        "longitude": lon,

        "road": road,

        "road_type": road_type,

        "blackspots": nearby_blackspots,

        "hazard_types": sorted(hazard_types),

        "raw_location": raw_location

    }