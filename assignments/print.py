from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENROUTER_API_KEY")

if key:
    print("API key loaded successfully")
else:
    print("API key NOT found")