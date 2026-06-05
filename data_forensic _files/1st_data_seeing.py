import json

candidates = []

with open("F:\\autorecruit-\\data_forensic _files\\candidates.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        candidates.append(json.loads(line))

import pprint

pprint.pprint(candidates[0])

# print(len(candidates))
# print(candidates[0])
# 
# print(json.dumps(candidates[0]["profile"]["summary"], indent=4))
# print(json.dumps(candidates[0]["career_history"], indent=4))
# print(json.dumps(list(candidates[0]["profile"]["summary"].keys()), indent=4))