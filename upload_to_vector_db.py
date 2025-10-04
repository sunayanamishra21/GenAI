import json
import requests
import numpy as np
from typing import List, Dict
import time

class VectorDBUploader:
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def upload_from_local_files(self, 
                               embeddings_file: str = "embeddings_data.json",
                               vectors_file: str = "embeddings_vectors.npy") -> bool:
        """Upload embeddings from local files to vector database"""
        
        try:
            # Load embeddings data
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            print(f"Loaded {len(embeddings_data)} embeddings from {embeddings_file}")
            
            # Try different vector database endpoints
            endpoints_to_try = [
                # Common vector database API endpoints
                "https://api.pinecone.io/v1/vectors/upsert",
                "https://api.weaviate.io/v1/objects",
                "https://api.vectordb.io/v1/vectors/upsert",
                "https://api.supabase.com/v1/vectors",
                "https://api.milvus.io/v1/vectors",
                "https://api.qdrant.io/collections/documents/points",
                # Generic endpoints that might work
                "https://api.vector-db.com/v1/upsert",
                "https://api.embeddings.com/v1/store",
            ]
            
            for endpoint in endpoints_to_try:
                if self.try_upload_to_endpoint(endpoint, embeddings_data):
                    return True
            
            print("Could not find a working vector database endpoint")
            return False
            
        except Exception as e:
            print(f"Error uploading embeddings: {e}")
            return False
    
    def try_upload_to_endpoint(self, endpoint: str, embeddings_data: List[Dict]) -> bool:
        """Try to upload to a specific endpoint"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            # Try different payload formats for different vector databases
            payload_formats = [
                # Format 1: Standard vector format
                {"vectors": embeddings_data},
                
                # Format 2: Pinecone format
                {"vectors": [{"id": v["id"], "values": v["vector"], "metadata": v["metadata"]} for v in embeddings_data]},
                
                # Format 3: Weaviate format
                {"objects": [{"class": "Document", "properties": v["metadata"], "vector": v["vector"]} for v in embeddings_data]},
                
                # Format 4: Qdrant format
                {"points": [{"id": v["id"], "vector": v["vector"], "payload": v["metadata"]} for v in embeddings_data]},
                
                # Format 5: Simple upsert format
                {"data": embeddings_data},
            ]
            
            for i, payload in enumerate(payload_formats):
                try:
                    print(f"Trying endpoint: {endpoint} with format {i+1}")
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=10
                    )
                    
                    if response.status_code in [200, 201, 202]:
                        print(f"SUCCESS! Uploaded to {endpoint} using format {i+1}")
                        print(f"Response: {response.status_code} - {response.text[:200]}...")
                        return True
                    else:
                        print(f"Failed with format {i+1}: {response.status_code} - {response.text[:200]}...")
                        
                except requests.exceptions.Timeout:
                    print(f"Timeout for format {i+1}")
                    continue
                except requests.exceptions.ConnectionError:
                    print(f"Connection error for format {i+1}")
                    break  # Don't try other formats if we can't connect
                except Exception as e:
                    print(f"Error with format {i+1}: {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"Error trying endpoint {endpoint}: {e}")
            return False
    
    def upload_with_custom_endpoint(self, base_url: str, collection_name: str = "documents") -> bool:
        """Upload to a custom vector database endpoint"""
        
        try:
            # Load embeddings data
            with open("embeddings_data.json", 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            print(f"Uploading {len(embeddings_data)} embeddings to {base_url}")
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            # Try different endpoint patterns
            endpoints = [
                f"{base_url}/v1/vectors/upsert",
                f"{base_url}/v1/collections/{collection_name}/vectors",
                f"{base_url}/api/v1/upsert",
                f"{base_url}/vectors",
                f"{base_url}/embeddings",
            ]
            
            for endpoint in endpoints:
                try:
                    response = requests.post(
                        endpoint,
                        headers=headers,
                        json={"vectors": embeddings_data},
                        timeout=30
                    )
                    
                    if response.status_code in [200, 201, 202]:
                        print(f"SUCCESS! Uploaded to {endpoint}")
                        print(f"Response: {response.status_code} - {response.text[:200]}...")
                        return True
                    else:
                        print(f"Failed at {endpoint}: {response.status_code} - {response.text[:200]}...")
                        
                except Exception as e:
                    print(f"Error at {endpoint}: {e}")
                    continue
            
            return False
            
        except Exception as e:
            print(f"Error uploading to custom endpoint: {e}")
            return False

def main():
    # Your API key
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xFcDgqa1ADs22JIIwFNvrN45FWcgRQQgqZ1AetfKoSU"
    
    uploader = VectorDBUploader(api_key)
    
    print("Vector Database Upload Script")
    print("=" * 40)
    print("1. Try automatic discovery of vector database")
    print("2. Upload to custom endpoint")
    print("3. Show embedding statistics")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        print("\nTrying to auto-discover vector database...")
        success = uploader.upload_from_local_files()
        
        if success:
            print("\n[SUCCESS] Embeddings uploaded successfully!")
        else:
            print("\n[INFO] Auto-discovery failed. You may need to specify the correct endpoint manually.")
            
    elif choice == "2":
        base_url = input("Enter the base URL of your vector database (e.g., https://api.yourvectordb.com): ").strip()
        if base_url:
            collection_name = input("Enter collection/index name (optional, default: documents): ").strip() or "documents"
            success = uploader.upload_with_custom_endpoint(base_url, collection_name)
            
            if success:
                print("\n[SUCCESS] Embeddings uploaded successfully!")
            else:
                print("\n[ERROR] Upload failed!")
        else:
            print("[ERROR] No URL provided")
            
    elif choice == "3":
        try:
            with open("embeddings_data.json", 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            print(f"\nEmbedding Statistics:")
            print(f"- Total chunks: {len(embeddings_data)}")
            print(f"- Vector dimensions: {len(embeddings_data[0]['vector']) if embeddings_data else 0}")
            print(f"- Source document: {embeddings_data[0]['metadata']['source'] if embeddings_data else 'N/A'}")
            
            # Show sample text
            if embeddings_data:
                sample_text = embeddings_data[0]['metadata']['text'][:200]
                print(f"\nSample text (first 200 chars):")
                print(f"'{sample_text}...'")
                
        except Exception as e:
            print(f"Error reading embedding statistics: {e}")
    
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
