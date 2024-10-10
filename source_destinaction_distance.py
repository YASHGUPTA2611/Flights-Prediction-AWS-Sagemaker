from geopy.distance import geodesic as GD
from geopy.geocoders import Nominatim
from functools import lru_cache
import pandas as pd
import json

geolocator = Nominatim(user_agent="MyApp")


train = pd.read_csv("train.csv")

sd_subset = train[["source", "destination"]]


# Use caching to store previously computed distances
@lru_cache(maxsize=None)
def get_lat_long(city):
    """Get latitude and longitude for a given city."""
    location = geolocator.geocode(f"{city}, India")
    if location:
        return location.latitude, location.longitude
    else:
        return None

def distance_between_cities(source, destination):
    """Calculate the distance between two cities using geopy."""
    lat_long_city1 = get_lat_long(source)
    lat_long_city2 = get_lat_long(destination)

    # Check if both cities returned valid coordinates
    if lat_long_city1 and lat_long_city2:
        distance = GD(lat_long_city1, lat_long_city2).km
        return distance
    else:
        return None  # Return None if any city has missing coordinates
    
def return_distance_dict(train):
    """Apply distance calculation to each row in the DataFrame."""
    distance = train.apply(lambda row: distance_between_cities(row['source'], row['destination']), axis=1)
    return pd.DataFrame(distance, columns=["distance_between_cities"])

def source_destination(train):
    sd_subset = train[["source","destination"]]
    sd_subset[["source", "destination"]] = sd_subset[["source", "destination"]].apply(lambda col: col.str.lower())
    sd_subset["source_destination"] = (sd_subset["source"].astype(str) + "_" + sd_subset["destination"])
    sd_subset["distance"] = sd_subset.apply(lambda row: distance_between_cities(row['source'], row['destination']), axis=1)
    
    dictionary  = {key : value for key,value in zip(sd_subset["source_destination"].values, sd_subset["distance"].values)}
    return dictionary

dictionary_distance = source_destination(sd_subset)

with open("dictionary_distance", 'w') as json_file:
    json.dump(dictionary_distance, json_file, indent=4) 

 