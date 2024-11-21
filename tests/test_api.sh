#!/bin/bash

echo "Testing the root endpoint"
curl http://localhost:8000/

echo "Testing the file upload endpoint"
curl -X 'POST' 'http://localhost:8000/uploadfile/' -F "file=@sources/obama.txt"

echo "Testing the ask question endpoint"
curl -X 'POST' 'http://localhost:8000/ask/' \
-H "Content-Type: application/json" \
-d '{
  "question": "Who is the president of the United States according to the context provided?",
  "document_id": 1
}'
