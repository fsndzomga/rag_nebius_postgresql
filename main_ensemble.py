from fastapi import (FastAPI, UploadFile, HTTPException,
                     Depends, BackgroundTasks)
import os
from pydantic import BaseModel
from dotenv import load_dotenv
import shutil
import io
from database import get_db, File, FileChunk
from sqlalchemy.orm import Session
from parsers import FileParser
from chunks import TextProcessor, client
from sqlalchemy import select
import logging

load_dotenv()

app = FastAPI()


class Question(BaseModel):
    question: str


class AskModel(BaseModel):
    document_id: int
    question: str


@app.get("/")
async def root(db: Session = Depends(get_db)):
    # Query the database for all files
    files_query = select(File)
    files = db.scalars(files_query).all()

    # Return the list of files
    files_list = [{"file_id": file.file_id, "file_name": file.file_name} for file in files]

    return files_list


@app.post("/uploadfile/")
async def upload_file(background_tasks: BackgroundTasks, file: UploadFile, db: Session = Depends(get_db), ):
    # Define allowed file extensions
    allowed_extensions = ["txt", "pdf"]

    # Check if the file extension is allowed
    file_extension = file.filename.split('.')[-1]
    if file_extension not in allowed_extensions:
        raise HTTPException(status_code=400, detail="File type not allowed")

    folder = "sources"
    try:
        # Ensure the directory exists
        os.makedirs(folder, exist_ok=True)

        # Secure way to save the file
        file_location = os.path.join(folder, file.filename)
        file_content = await file.read()  # Read file content as bytes

        # Remove null bytes
        clean_file_content = file_content.replace(b'\x00', b'')

        with open(file_location, "wb+") as file_object:
            # Convert bytes content to a file-like object
            file_like_object = io.BytesIO(clean_file_content)
            # Use shutil.copyfileobj for secure file writing
            shutil.copyfileobj(file_like_object, file_object)

        # Parse the file content
        content_parser = FileParser(filepath=file_location)
        file_text_content = content_parser.parse()

        # Save the file content to the database
        new_file = File(file_name=file.filename, file_content=file_text_content)
        db.add(new_file)
        db.commit()
        db.refresh(new_file)

        # add background task to process the text
        text_processor = TextProcessor(db=db, file_id=new_file.file_id)
        background_tasks.add_task(text_processor.chunk_and_embed, file_text_content)

        return {"info": "File saved", "filename": file.filename}

    except Exception as e:
        # Log the exception (add actual logging in production code)
        print(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Error saving file")


# Function to get similar chunks
async def get_similar_chunks(file_id: int, question: str, db: Session, limit=5):
    try:
        # create the embeddings for the question
        response = client.embeddings.create(
            model="BAAI/bge-en-icl",
            input=question,
            encoding_format="float"
        )

        question_embedding = response.data[0].embedding

        # Query the database for chunks
        similar_chunks_query = select(FileChunk).where(FileChunk.file_id == file_id).order_by(FileChunk.embedding_vector.l2_distance(question_embedding)).limit(limit)

        similar_chunks = db.scalars(similar_chunks_query).all()

        return similar_chunks
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/")
async def ask_question(request: AskModel, db: Session = Depends(get_db)):
    """
    Handles user questions by fetching relevant document chunks,
    generating responses using multiple models, and synthesizing a final answer.
    """
    # Check if the NEBIUS API key is set
    if os.environ.get("NEBIUS_API_KEY") is None:
        raise HTTPException(status_code=500, detail="NEBIUS API key not set")

    try:
        # Fetch similar chunks from the database
        similar_chunks = await get_similar_chunks(request.document_id, request.question, db)
        second_similar_chunks = await get_similar_chunks(request.document_id, request.question, db, limit=10)

        # Combine the chunks into a single context for the AI models
        context = " ".join(chunk.chunk_text for chunk in similar_chunks)
        second_context = " ".join(chunk.chunk_text for chunk in second_similar_chunks)

        logging.info(f"Primary Context: {context}")
        logging.info(f"Secondary Context: {second_context}")

        # Create system messages for the models
        system_message = (
            f"You are a helpful assistant. Here is the context to use to reply to the user question: {context}"
        )
        second_system_message = (
            f"You are a helpful assistant. Here is the context to use to reply to the user question: {second_context}"
        )

        # Generate responses from two different models
        primary_response = client.chat.completions.create(
            model="Qwen/Qwen2.5-Coder-32B-Instruct",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": request.question}
            ]
        )
        secondary_response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-405B-Instruct",
            messages=[
                {"role": "system", "content": second_system_message},
                {"role": "user", "content": request.question}
            ]
        )

        # Synthesize a final response based on the two model outputs
        final_system_message = (
            f"You are a helpful assistant. Based on this first context: {context} and this second context: {second_context}, "
            f"here is the first possible response: {primary_response.choices[0].message.content} "
            f"and here is the second possible response: {secondary_response.choices[0].message.content}. "
            f"Based on that, craft a final response to the user question."
        )

        final_response = client.chat.completions.create(
            model="nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
            messages=[
                {"role": "system", "content": final_system_message},
                {"role": "user", "content": request.question}
            ]
        )

        # Return the final response
        return {"response": final_response.choices[0].message.content}

    except Exception as e:
        # Handle errors and provide appropriate feedback
        logging.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/find-similar-chunks/{file_id}")
async def find_similar_chunks(file_id: int, question: Question, db: Session = Depends(get_db)):
    try:
        similar_chunks = await get_similar_chunks(file_id, question.question, db)

        formated_response = [{"chunk_id": chunk.chunk_id, "chunk_text": chunk.chunk_text} for chunk in similar_chunks]

        return formated_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
