import requests

GROQ_API_KEY = "********************" # Insert your Groq key here

headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
res = requests.get("https://api.groq.com/openai/v1/models", headers=headers)

if res.status_code == 200:
    for m in res.json().get("data", []):
        print(m["id"])
else:
    print(f"Error {res.status_code}: {res.text}")