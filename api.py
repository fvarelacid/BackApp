#### Fetch sounds from Freedsound API ####

import requests
import json
import freesound
import math

api_key = "gFX59R3ZwGkUeOHLHyqXK9UeuOyUBsrqDrg2uCIf"

freesound_client = freesound.FreesoundClient()
freesound_client.set_token(api_key)

# response_sounds = requests.get("https://freesound.org/apiv2/search/text/?query=agony&filter=tag:agony&token=" + api_key)

# response_sounds_json = response_sounds.json()
# print(response_sounds_json)

# response_sounds_json_results = response_sounds_json["results"]
# print(len(response_sounds_json_results))

# for result in response_sounds_json_results:
#     print(result["id"])

print("Searching for 'agony':")
print("----------------------------")
results_pager = freesound_client.text_search(
    query="agony",
    filter="tag:agony",
    sort="rating_desc",
    fields="id,name,avg_rating,download"
)

num_results = results_pager.count
num_pages = int(math.ceil(num_results / 15))

print("Num results:", results_pager.count)
print("Num pages:", num_pages)

print("\t----- PAGE 1 -----")
count_sound = 0
for sound in results_pager:
    count_sound += 1
    print("\t-", sound.name, "rated", sound.avg_rating, "download", sound.download)
print(count_sound)
print("\t----- PAGE 2 -----")
results_pager = results_pager.next_page()
for sound in results_pager:
    print("\t-", sound.name, "rated", sound.avg_rating)
print()

# for pages in results_pager.count:
