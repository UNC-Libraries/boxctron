import json

def from_json(filepath):
  with open(filepath, 'r') as fp:
    data = json.load(fp)
  return data

def to_json(x, filepath):
  with open(filepath, 'w') as fp:
    json.dump(x, fp, indent=2)