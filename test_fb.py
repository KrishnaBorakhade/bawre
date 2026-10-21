import os
from dotenv import load_dotenv

load_dotenv()
from firebase_module import _get_firebase_app, upload_graph_to_firebase

print("Testing Firebase connection...")
bucket, db = _get_firebase_app()
if bucket and db:
    print(f"Success! Connected to bucket: {bucket.name}")
else:
    print("Failed to initialize Firebase.")
