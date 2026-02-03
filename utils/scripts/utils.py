import json

def dbg(msg):
    print(f"[DEBUG] {msg}", flush=True)

# Load config
with open('../config.json', 'r') as f:
    config = json.load(f)