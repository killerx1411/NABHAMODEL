# Run once in Python to fix your existing metadata.json
import json

with open("model/metadata.json", "r", encoding="utf-8-sig", errors="replace") as f:
    data = json.load(f)

with open("model/metadata.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=True)  # ensure_ascii=True avoids all non-ASCII