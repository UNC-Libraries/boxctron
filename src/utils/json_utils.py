import json

def from_json(filepath):
  with open(filepath, 'r') as fp:
    data = json.load(fp)
  return data
