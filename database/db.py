from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import text
from sqlalchemy_utils import database_exists, create_database
from pgvector.sqlalchemy import Vector
from dotenv import load_dotenv
import urllib.parse
import os

# Load environment variables from .env
load_dotenv()

# Database credentials
POSTGRES_USERNAME = os.getenv("POSTGRES_USERNAME")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
DATABASE_NAME = os.getenv("DATABASE_NAME", "fastapi_rag_noframework_db")
encoded_password = urllib.parse.quote_plus(POSTGRES_PASSWORD)

# Construct the database URL for SQLAlchemy
database_url = (
    f"postgresql+psycopg2://{POSTGRES_USERNAME}:{encoded_password}@"
    f"{POSTGRES_HOST}:{POSTGRES_PORT}/{DATABASE_NAME}"
)

# Create the SQLAlchemy engine
engine = create_engine(database_url)

# Create the database if it doesn't exist
if not database_exists(engine.url):
    try:
        create_database(engine.url)
        print(f"Database {DATABASE_NAME} created successfully.")
    except Exception as e:
        print(f"Error creating database: {e}")

# Create a session
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Define the models
Base = declarative_base()


class File(Base):
    __tablename__ = "files"

    file_id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String(255), nullable=False)
    file_content = Column(Text, nullable=False)


class FileChunk(Base):
    __tablename__ = "file_chunks"

    chunk_id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("files.file_id"), nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding_vector = Column(Vector(4096))


# Ensure that the vector extension is enabled in the database
with engine.begin() as connection:
    try:
        connection.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        print("Vector extension ensured.")
    except Exception as e:
        print(f"Error enabling vector extension: {e}")

# Create the tables
try:
    Base.metadata.create_all(engine)
    print("Tables created successfully.")
except Exception as e:
    print(f"Error while creating tables: {e}")
