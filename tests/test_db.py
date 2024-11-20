import sys
import os

# Add the parent directory to the sys.path to import the local module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database import engine, SessionLocal

# Attempt to connect to the database and create a session
try:
    with engine.connect() as connection:
        print("Connected to the database successfully!")
        session = SessionLocal()
        print("Session created successfully!")
except Exception as e:
    print(f"Error connecting to the database: {e}")
