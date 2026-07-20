import json
with open('backend/rag/rca_metadata.json', 'r') as f:
    data = json.load(f)
print(f'Sites: {len(data.keys())}')