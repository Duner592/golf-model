import json
from datagolf_client import DataGolfClient

BASE_URL = "https://feeds.datagolf.com"
ENDPOINT_UPCOMING = (
    "get-schedule"  # Replace with the actual path from the DataGolf docs
)

dg = DataGolfClient(BASE_URL)
upcoming = dg.get(ENDPOINT_UPCOMING, params={"tour": "pga"})

# Print the top-level keys in the response
print("Top-level keys in response:", upcoming.keys())

# Pretty-print first few records for a human-inspectable sample
# (You can adjust this or remove slicing as appropriate)
if isinstance(upcoming, dict):
    key = next(iter(upcoming))
    if isinstance(upcoming[key], list):
        print(f"Sample records for '{key}':")
        print(json.dumps(upcoming[key][:2], indent=2))
    else:
        print(json.dumps(upcoming, indent=2))

# Write the FULL pretty-printed JSON to file
with open("upcoming-events.json", "w", encoding="utf-8") as f:
    json.dump(upcoming, f, indent=2, ensure_ascii=False)

print("Saved pretty-formatted output to upcoming-events.json")
