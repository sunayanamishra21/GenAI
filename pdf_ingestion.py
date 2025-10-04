import os
import PyPDF2
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict
import time

class PDFIngestionService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db_url = "https://api.vectordb.io/v1"  # Adjust based on your vector database provider
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from PDF file"""
        print(f"Extracting text from {pdf_path}...")
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
                
        print(f"Extracted {len(text)} characters from PDF")
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks for better embedding"""
        print(f"Chunking text into segments of {chunk_size} characters...")
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                boundary = max(last_period, last_newline)
                
                if boundary > start + chunk_size // 2:  # Only if boundary is reasonable
                    chunk = chunk[:boundary + 1]
                    end = start + boundary + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
        print(f"Created {len(chunks)} text chunks")
        return chunks
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for text chunks"""
        print(f"Creating embeddings for {len(texts)} text chunks...")
        
        embeddings = self.embedding_model.encode(texts)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        
        return embeddings.tolist()
    
    def store_in_vector_db(self, texts: List[str], embeddings: List[List[float]], metadata: Dict = None) -> bool:
        """Store embeddings in vector database"""
        print(f"Storing {len(embeddings)} embeddings in vector database...")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare data for vector database
        vectors_data = []
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            vector_data = {
                "id": f"doc_{i}_{int(time.time())}",
                "vector": embedding,
                "metadata": {
                    "text": text,
                    "chunk_id": i,
                    "source": "CELEX_32016R0679_EN_TXT.pdf",
                    **(metadata or {})
                }
            }
            vectors_data.append(vector_data)
        
        # Note: The actual API endpoint and format may vary based on your vector database provider
        # This is a generic example - you may need to adjust the endpoint and payload format
        try:
            response = requests.post(
                f"{self.vector_db_url}/vectors/upsert",
                headers=headers,
                json={"vectors": vectors_data}
            )
            
            if response.status_code in [200, 201]:
                print("Successfully stored embeddings in vector database")
                return True
            else:
                print(f"Failed to store embeddings: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to vector database: {e}")
            return False
    
    def ingest_pdf(self, pdf_path: str) -> bool:
        """Complete pipeline to ingest PDF into vector database"""
        try:
            # Step 1: Extract text from PDF
            text = self.extract_text_from_pdf(pdf_path)
            
            if not text.strip():
                print("No text found in PDF")
                return False
            
            # Step 2: Chunk the text
            chunks = self.chunk_text(text)
            
            # Step 3: Create embeddings
            embeddings = self.create_embeddings(chunks)
            
            # Step 4: Store in vector database
            success = self.store_in_vector_db(chunks, embeddings)
            
            if success:
                print(f"Successfully ingested {pdf_path} into vector database")
                print(f"- Extracted {len(text)} characters")
                print(f"- Created {len(chunks)} chunks")
                print(f"- Generated {len(embeddings)} embeddings")
            
            return success
            
        except Exception as e:
            print(f"Error during PDF ingestion: {e}")
            return False

def main():
    # Your API key
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xFcDgqa1ADs22JIIwFNvrN45FWcgRQQgqZ1AetfKoSU"
    
    # PDF file path
    pdf_path = "CELEX_32016R0679_EN_TXT.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    # Initialize ingestion service
    ingestion_service = PDFIngestionService(api_key)
    
    # Ingest PDF
    success = ingestion_service.ingest_pdf(pdf_path)
    
    if success:
        print("\n[SUCCESS] PDF ingestion completed successfully!")
    else:
        print("\n[ERROR] PDF ingestion failed!")

if __name__ == "__main__":
    main()
