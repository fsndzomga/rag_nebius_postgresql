# **FastAPI RAG with PostgreSQL and Nebius**

<img width="1364" alt="Screenshot 2024-11-20 at 13 02 01" src="https://github.com/user-attachments/assets/d3f9d33b-fd43-4d98-9c04-c0df3019ca0b">


## **Overview**
This project demonstrates a **Retrieval-Augmented Generation (RAG)** implementation using **FastAPI** as the web framework, **PostgreSQL** (managed by **Nebius**) as the database, and vector search for chunk similarity. It processes uploaded files (text or PDF), extracts content, embeds the content using vector embeddings, and provides endpoints to ask questions based on the embedded content.

## **Features**
- **File Upload**: Upload text (`.txt`) or PDF (`.pdf`) files to the server.
- **Content Extraction**: Extracts text content from files, with OCR fallback for PDFs without readable text.
- **Database Storage**: Saves extracted content and file metadata in PostgreSQL.
- **Vector Embedding**: Uses vector embeddings for content similarity search.
- **Ask Questions**: Query the database for context-aware answers.
- **API Endpoints**: Includes endpoints for uploading files, querying context, and retrieving similar chunks.

---

## **Tech Stack**
- **Backend**: [FastAPI](https://fastapi.tiangolo.com/)
- **Database**: [PostgreSQL](https://www.postgresql.org/) (Managed by [Nebius](https://nebius.com))
- **Vector Embedding**: pgvector for vector similarity search.
- **PDF/Text Parsing**:
  - [PyPDF2](https://pypi.org/project/PyPDF2/) for PDF parsing.
  - [pytesseract](https://pypi.org/project/pytesseract/) for OCR.
- **Environment Management**: [dotenv](https://pypi.org/project/python-dotenv/)
- **Embedding Model**: Integrated with vector models like `BAAI/bge-en-icl`.

---

## **Prerequisites**
- Python 3.8+
- PostgreSQL with pgvector extension enabled.
- Installed Tesseract OCR (for OCR capabilities).
- `.env` file for configuration.

### Example `.env` File:
```env
POSTGRES_USERNAME=your_postgres_username
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_HOST=your_postgres_host
POSTGRES_PORT=5432
DATABASE_NAME=fastapi_rag_noframework_db

NEBIUS_API_KEY=your_nebius_api_key
```

---

## **Setup Instructions**

### **1. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### **2. Set Up a Virtual Environment**
```bash
python3 -m venv venv
source venv/bin/activate
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **4. Configure PostgreSQL**
Ensure PostgreSQL is running and `pgvector` extension is installed. The database will be created automatically if it does not exist.

---

## **Usage**

### **1. Start the Server**
Run the FastAPI app:
```bash
uvicorn main:app --reload
```

### **2. API Endpoints**
| Endpoint                  | Method | Description                                                  |
|---------------------------|--------|--------------------------------------------------------------|
| `/`                       | GET    | Lists all uploaded files.                                    |
| `/uploadfile/`            | POST   | Upload a file (text or PDF).                                 |
| `/ask/`                   | POST   | Ask a question using the document's context.                 |
| `/find-similar-chunks/{file_id}` | POST   | Retrieve similar content chunks for a file and question.     |

#### **Example Request: Upload File**
```bash
curl -X 'POST' 'http://localhost:8000/uploadfile/' -F "file=@sources/obama.txt"
```

#### **Example Request: Ask Question**
```bash
curl -X 'POST' 'http://localhost:8000/ask/' \
-H "Content-Type: application/json" \
-d '{
  "question": "Who is the president of the United States according to the context provided?",
  "document_id": 1
}'
```

---

## **File Parsing**

### **Supported File Types**
- `.txt`
- `.pdf` (with OCR fallback for scanned PDFs)

### **Parser Workflow**
1. **Text Extraction**:
   - Plain text files: Simple file reading.
   - PDFs: Uses `PyPDF2` for text-based PDFs and `pytesseract` for OCR.
2. **Repair and Validation**:
   - Repairs malformed PDFs using `pikepdf`.
   - Validates PDF structure before parsing.

---

## **Database**

### **Tables**
1. **`files`**:
   - Stores file metadata and content.
   - Columns: `file_id`, `file_name`, `file_content`.
2. **`file_chunks`**:
   - Stores content chunks with vector embeddings.
   - Columns: `chunk_id`, `file_id`, `chunk_text`, `embedding_vector`.

### **Deleting Tables**
The project includes a utility to drop all tables:
```bash
python delete_tables.py
```

---

## **Testing**
The project includes automated test scripts:
```bash
python tests/test_parsers.py
```

These tests are not written with best practices in mind but were just create to quickly see if the main functionalities are working.

---

## **Future Enhancements**
- Add support for additional file types (e.g., Word documents).
- Implement authentication for secure API usage.
- Enhance chunking logic for better context extraction.

---

## **Contributing**
Contributions are welcome! Please fork the repository and submit a pull request.

---

## **License**
This project is licensed under the MIT License. See the LICENSE file for details.
