#### Fetch sounds from Freedsound API ####

import requests
import json

api_key = "gFX59R3ZwGkUeOHLHyqXK9UeuOyUBsrqDrg2uCIf"

response_sounds = requests.get("https://freesound.org/apiv2/search/text/?query=agony&filter=tag:agony&token=" + api_key)

response_sounds_json = response_sounds.json()
print(response_sounds_json)

response_sounds_json_results = response_sounds_json["results"]
print(len(response_sounds_json_results))

for result in response_sounds_json_results:
    print(result["id"])