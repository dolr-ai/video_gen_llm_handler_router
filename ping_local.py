# Test the token with a sample request (you need the server running)

import os
import requests
from dotenv import load_dotenv
load_dotenv()

token = os.environ.get("JWT_TOKEN")

# Uncomment and run this if your server is running on localhost:8000
# """
headers = {"Authorization": f"Bearer {token}"}
response = requests.get("http://localhost:8000/v1/status", headers=headers)
print(f"Server response: {response.status_code}")
if response.status_code == 200:
    print("✅ Authentication successful!")
    print(response.json())
else:
    print("❌ Authentication failed:")
    print(response.text)
# """

print("To test this token:")
print("1. Make sure your server is running")
print("2. Set the PUBLIC_KEY_TO_VERIFY_INCOMING_CALLS environment variable")
print("3. Uncomment and run the request code above")
print(f"\nToken to use: {token}")
