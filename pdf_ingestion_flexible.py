import os
import PyPDF2
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
import json
from typing import List, Dict, Optional
import time
import base64

class VectorDBProvider:
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def store_vectors(self, vectors_data: List[Dict]) -> bool:
        """Store vectors in the vector database - to be implemented by specific providers"""
        raise NotImplementedError("This method should be implemented by specific vector database providers")

class PineconeProvider(VectorDBProvider):
    def __init__(self, api_key: str):
        super().__init__(api_key)
        self.base_url = "https://api.pinecone.io"
    
    def store_vectors(self, vectors_data: List[Dict]) -> bool:
        """Store vectors in Pinecone"""
        headers = {
            "Api-Key": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            # For Pinecone, you need to specify the index name
            index_name = "your-index-name"  # You'll need to provide this
            response = requests.post(
                f"{self.base_url}/vectors/upsert",
                headers=headers,
                json={
                    "vectors": vectors_data
                }
            )
            
            if response.status_code in [200, 201]:
                print("Successfully stored embeddings in Pinecone")
                return True
            else:
                print(f"Failed to store embeddings in Pinecone: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Pinecone: {e}")
            return False

class WeaviateProvider(VectorDBProvider):
    def __init__(self, api_key: str, url: str = None):
        super().__init__(api_key)
        self.base_url = url or "http://localhost:8080"
    
    def store_vectors(self, vectors_data: List[Dict]) -> bool:
        """Store vectors in Weaviate"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Weaviate requires a schema first, this is a simplified example
            for vector_data in vectors_data:
                response = requests.post(
                    f"{self.base_url}/v1/objects",
                    headers=headers,
                    json={
                        "class": "Document",
                        "properties": vector_data.get("metadata", {}),
                        "vector": vector_data.get("vector", [])
                    }
                )
                
                if response.status_code not in [200, 201]:
                    print(f"Failed to store vector in Weaviate: {response.status_code} - {response.text}")
                    return False
            
            print("Successfully stored embeddings in Weaviate")
            return True
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to Weaviate: {e}")
            return False

class ChromaProvider(VectorDBProvider):
    def __init__(self, api_key: str = None, url: str = None):
        super().__init__(api_key or "dummy")
        self.base_url = url or "http://localhost:8000"
    
    def store_vectors(self, vectors_data: List[Dict]) -> bool:
        """Store vectors in ChromaDB"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key != "dummy":
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            response = requests.post(
                f"{self.base_url}/api/v1/collections/documents/upsert",
                headers=headers,
                json={
                    "documents": [v.get("metadata", {}).get("text", "") for v in vectors_data],
                    "embeddings": [v.get("vector", []) for v in vectors_data],
                    "metadatas": [v.get("metadata", {}) for v in vectors_data],
                    "ids": [v.get("id", f"doc_{i}") for i, v in enumerate(vectors_data)]
                }
            )
            
            if response.status_code in [200, 201]:
                print("Successfully stored embeddings in ChromaDB")
                return True
            else:
                print(f"Failed to store embeddings in ChromaDB: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to ChromaDB: {e}")
            return False

class GenericVectorDBProvider(VectorDBProvider):
    def __init__(self, api_key: str, base_url: str):
        super().__init__(api_key)
        self.base_url = base_url
    
    def store_vectors(self, vectors_data: List[Dict]) -> bool:
        """Store vectors in a generic vector database"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/vectors/upsert",
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

class PDFIngestionService:
    def __init__(self, vector_db_provider: VectorDBProvider):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_db_provider = vector_db_provider
        
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
        """Store embeddings in vector database using the configured provider"""
        print(f"Storing {len(embeddings)} embeddings in vector database...")
        
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
        
        return self.vector_db_provider.store_vectors(vectors_data)
    
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

def create_vector_db_provider(provider_type: str, api_key: str, **kwargs) -> VectorDBProvider:
    """Factory function to create appropriate vector database provider"""
    if provider_type.lower() == "pinecone":
        return PineconeProvider(api_key)
    elif provider_type.lower() == "weaviate":
        return WeaviateProvider(api_key, kwargs.get('url'))
    elif provider_type.lower() == "chroma":
        return ChromaProvider(api_key, kwargs.get('url'))
    elif provider_type.lower() == "generic":
        return GenericVectorDBProvider(api_key, kwargs.get('base_url'))
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

def main():
    # Your API key
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xFcDgqa1ADs22JIIwFNvrN45FWcgRQQgqZ1AetfKoSU"
    
    # PDF file path
    pdf_path = "CELEX_32016R0679_EN_TXT.pdf"
    
    # Check if PDF exists
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print("Available vector database providers:")
    print("1. Pinecone")
    print("2. Weaviate")
    print("3. ChromaDB")
    print("4. Generic Vector DB")
    
    # For demonstration, let's try with ChromaDB first (local)
    try:
        print("\nTrying ChromaDB (local) first...")
        vector_db_provider = ChromaProvider()
        ingestion_service = PDFIngestionService(vector_db_provider)
        
        # Ingest PDF
        success = ingestion_service.ingest_pdf(pdf_path)
        
        if success:
            print("\n[SUCCESS] PDF ingestion completed successfully with ChromaDB!")
        else:
            print("\n[INFO] ChromaDB not available, trying to save embeddings locally...")
            # Save embeddings to local file as backup
            save_embeddings_locally(pdf_path, api_key)
            
    except Exception as e:
        print(f"\n[ERROR] ChromaDB failed: {e}")
        print("Saving embeddings locally as backup...")
        save_embeddings_locally(pdf_path, api_key)

def save_embeddings_locally(pdf_path: str, api_key: str):
    """Save embeddings to local files as backup"""
    print("Creating embeddings and saving locally...")
    
    # Create a dummy provider that saves to files
    class LocalFileProvider(VectorDBProvider):
        def store_vectors(self, vectors_data: List[Dict]) -> bool:
            try:
                # Save as JSON
                with open('embeddings_data.json', 'w', encoding='utf-8') as f:
                    json.dump(vectors_data, f, indent=2, ensure_ascii=False)
                
                # Save as numpy array
                vectors = np.array([v['vector'] for v in vectors_data])
                np.save('embeddings_vectors.npy', vectors)
                
                # Save text chunks
                texts = [v['metadata']['text'] for v in vectors_data]
                with open('text_chunks.txt', 'w', encoding='utf-8') as f:
                    for i, text in enumerate(texts):
                        f.write(f"=== CHUNK {i} ===\n")
                        f.write(text)
                        f.write("\n\n")
                
                print("Successfully saved embeddings locally:")
                print("- embeddings_data.json (full data)")
                print("- embeddings_vectors.npy (vectors only)")
                print("- text_chunks.txt (text chunks)")
                return True
                
            except Exception as e:
                print(f"Error saving embeddings locally: {e}")
                return False
    
    try:
        vector_db_provider = LocalFileProvider(api_key)
        ingestion_service = PDFIngestionService(vector_db_provider)
        success = ingestion_service.ingest_pdf(pdf_path)
        
        if success:
            print("\n[SUCCESS] PDF embeddings saved locally!")
        else:
            print("\n[ERROR] Failed to save embeddings locally!")
            
    except Exception as e:
        print(f"\n[ERROR] Local save failed: {e}")

if __name__ == "__main__":
    main()
