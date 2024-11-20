from sqlalchemy.orm import Session
from database import FileChunk
from openai import OpenAI
from dotenv import load_dotenv
import os
import re

load_dotenv()

client = OpenAI(
        base_url="https://api.studio.nebius.ai/v1/",
        api_key=os.environ.get("NEBIUS_API_KEY"),
)


class TextProcessor:
    def __init__(self, db: Session, file_id: int, chunk_size: int = 3):
        self.db = db
        self.file_id = file_id
        self.chunk_size = chunk_size

    def sentence_tokenize(self, text: str):
        "Function to tokenize text into sentences"
        # Split text into sentences based on punctuation (.!?) using regex
        sentences = re.split(r"[.!?]", text)

        # Remove empty sentences
        sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

        return sentences

    def chunk_and_embed(self, text: str):
        # split text into sentences
        sentences = self.sentence_tokenize(text)

        # chunk sentences into groups of size chunk_size
        chunks = [' '.join(sentences[i:i + self.chunk_size]) for i in range(0, len(sentences), self.chunk_size)]

        # embed chunks
        for chunk in chunks:
            response = client.embeddings.create(
                model="BAAI/bge-en-icl",
                input=chunk,
                encoding_format="float"
            )

            embeddings = response.data[0].embedding

            # Save the chunk and its embedding to the database
            file_chunk = FileChunk(file_id=self.file_id, chunk_text=chunk, embedding_vector=embeddings)

            self.db.add(file_chunk)
        self.db.commit()
