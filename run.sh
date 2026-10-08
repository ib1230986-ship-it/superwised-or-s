#!/bin/bash
cd "$(dirname "$0")"
echo "Server start ho raha hai... browser mein http://localhost:8000 kholein"
python3 -m uvicorn main:app --port 8000
