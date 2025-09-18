# Test the token with a sample request against production

import os
import requests
from dotenv import load_dotenv
load_dotenv()

token = os.environ.get("JWT_TOKEN")

headers = {"Authorization": f"Bearer {token}"}
response = requests.get("https://video-gen-llm-handler-router.fly.dev/v1/status", headers=headers)
print(f"Server response: {response.status_code}")
if response.status_code == 200:
    print("✅ Authentication successful!")
    print(response.json())
else:
    print("❌ Authentication failed:")
    print(response.text)

print(f"\nToken to use: {token}")
