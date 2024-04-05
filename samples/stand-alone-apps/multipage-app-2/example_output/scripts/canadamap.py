import folium
import webbrowser

# Create a map object
m = folium.Map(location=[60.1087, -113.6426], zoom_start=3)

# Add markers for each city
locations = {
    'Alberta': [53.5461, -113.4938],
    'British Columbia': [48.4284, -123.3656],
    'Manitoba': [49.8951, -97.1384],
    'New Brunswick': [45.9636, -66.6431],
    'Newfoundland and Labrador': [47.5615, -52.7126],
    'Nova Scotia': [44.6488, -63.5752],
    'Ontario': [43.6511, -79.3832],
    'Prince Edward Island': [46.2382, -63.1311],
    'Quebec': [46.8139, -71.2080],
    'Saskatchewan': [50.4452, -104.6189],
    'Northwest Territories': [62.4540, -114.3718],
    'Nunavut': [63.7467, -68.5170],
    'Yukon': [60.7212, -135.0568]
}

location_list = [ {"province": "Alberta", "city": "Edmonton", "latitude": 53.5461, "longitude": -113.4938},
{"province": "British Columbia", "city": "Victoria", "latitude": 48.4284, "longitude": -123.3656},
{"province": "Manitoba", "city": "Winnipeg", "latitude": 49.8951, "longitude": -97.1384},
{"province": "New Brunswick", "city": "Fredericton", "latitude": 45.9636, "longitude": -66.6431},
{"province": "Newfoundland and Labrador", "city": "St. John's", "latitude": 47.5615, "longitude": -52.7126},
{"province": "Nova Scotia", "city": "Halifax", "latitude": 44.6488, "longitude": -63.5752},
{"province": "Ontario", "city": "Toronto", "latitude": 43.6511, "longitude": -79.3832},
{"province": "Prince Edward Island", "city": "Charlottetown", "latitude": 46.2382, "longitude": -63.1311},
{"province": "Quebec", "city": "Quebec City", "latitude": 46.8139, "longitude": -71.2080},
{"province": "Saskatchewan", "city": "Regina", "latitude": 50.4452, "longitude": -104.6189},
{"province": "Northwest Territories", "city": "Yellowknife", "latitude": 62.4540, "longitude": -114.3718},
{"province": "Nunavut", "city": "Iqaluit", "latitude": 63.7467, "longitude": -68.5170},
{"province": "Yukon", "city": "Whitehorse", "latitude": 60.7212, "longitude": -135.0568} 
]

for item in location_list:
    city = item["city"]
    location = [item["latitude"], item["longitude"]]
    folium.Marker(location=location, popup=city).add_to(m)

m.save("canada_map_2.html")
webbrowser.open("canada_map_2.html")