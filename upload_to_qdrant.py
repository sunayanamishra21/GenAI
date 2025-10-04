import json
import requests
import numpy as np
from typing import List, Dict
import time
import uuid

class QdrantUploader:
    def __init__(self, api_key: str, cluster_url: str):
        self.api_key = api_key
        self.cluster_url = cluster_url.rstrip('/')
        self.collection_name = "documents"  # Default collection name
        
    def upload_embeddings(self, embeddings_file: str = "embeddings_data.json") -> bool:
        """Upload embeddings to Qdrant vector database"""
        
        try:
            # Load embeddings data
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings_data = json.load(f)
            
            print(f"Loaded {len(embeddings_data)} embeddings from {embeddings_file}")
            print(f"Uploading to Qdrant cluster: {self.cluster_url}")
            
            # Step 1: Create collection if it doesn't exist
            if not self.create_collection():
                print("Failed to create collection")
                return False
            
            # Step 2: Upload points
            success = self.upload_points(embeddings_data)
            
            if success:
                print(f"\n[SUCCESS] Uploaded {len(embeddings_data)} embeddings to Qdrant!")
                return True
            else:
                print("\n[ERROR] Failed to upload embeddings to Qdrant")
                return False
                
        except Exception as e:
            print(f"Error uploading to Qdrant: {e}")
            return False
    
    def create_collection(self) -> bool:
        """Create collection in Qdrant if it doesn't exist"""
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["api-key"] = self.api_key
        
        # Collection configuration
        collection_config = {
            "vectors": {
                "size": 384,  # Dimension of our embeddings
                "distance": "Cosine"  # Distance metric
            },
            "optimizers_config": {
                "default_segment_number": 2
            },
            "replication_factor": 1
        }
        
        try:
            # Check if collection exists
            response = requests.get(
                f"{self.cluster_url}/collections/{self.collection_name}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                print(f"Collection '{self.collection_name}' already exists")
                return True
            
            # Create collection if it doesn't exist
            print(f"Creating collection '{self.collection_name}'...")
            response = requests.put(
                f"{self.cluster_url}/collections/{self.collection_name}",
                headers=headers,
                json=collection_config,
                timeout=10
            )
            
            if response.status_code in [200, 201]:
                print(f"Successfully created collection '{self.collection_name}'")
                return True
            else:
                print(f"Failed to create collection: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error creating collection: {e}")
            return False
    
    def upload_points(self, embeddings_data: List[Dict]) -> bool:
        """Upload points to Qdrant collection"""
        
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["api-key"] = self.api_key
        
        # Convert embeddings to Qdrant format
        points = []
        for i, data in enumerate(embeddings_data):
            point = {
                "id": str(uuid.uuid4()),  # Generate UUID for each point
                "vector": data["vector"],
                "payload": {
                    "text": data["metadata"]["text"],
                    "chunk_id": data["metadata"]["chunk_id"],
                    "source": data["metadata"]["source"],
                    "original_id": data.get("id", f"doc_{i}")  # Keep original ID in payload
                }
            }
            points.append(point)
        
        # Upload in batches to avoid timeouts
        batch_size = 100
        total_batches = (len(points) + batch_size - 1) // batch_size
        
        print(f"Uploading {len(points)} points in {total_batches} batches...")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(points))
            batch_points = points[start_idx:end_idx]
            
            try:
                response = requests.put(
                    f"{self.cluster_url}/collections/{self.collection_name}/points",
                    headers=headers,
                    json={"points": batch_points},
                    timeout=30
                )
                
                if response.status_code in [200, 201]:
                    print(f"Batch {batch_idx + 1}/{total_batches}: Uploaded {len(batch_points)} points")
                else:
                    print(f"Batch {batch_idx + 1}/{total_batches}: Failed - {response.status_code} - {response.text}")
                    return False
                    
            except Exception as e:
                print(f"Batch {batch_idx + 1}/{total_batches}: Error - {e}")
                return False
        
        return True
    
    def verify_upload(self) -> bool:
        """Verify the upload by checking collection info"""
        
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            # Get collection info
            response = requests.get(
                f"{self.cluster_url}/collections/{self.collection_name}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                collection_info = response.json()
                points_count = collection_info.get("result", {}).get("points_count", 0)
                print(f"Collection '{self.collection_name}' now contains {points_count} points")
                return True
            else:
                print(f"Failed to verify upload: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error verifying upload: {e}")
            return False

def main():
    # Your configuration
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xFcDgqa1ADs22JIIwFNvrN45FWcgRQQgqZ1AetfKoSU"
    cluster_url = "https://02ac6bb1-cdd6-4187-8bdc-f7f82d20ff0d.us-east4-0.gcp.cloud.qdrant.io"
    
    print("Qdrant Vector Database Upload")
    print("=" * 40)
    print(f"Cluster URL: {cluster_url}")
    print(f"Collection: documents")
    print(f"API Key: {'*' * 20}...{api_key[-10:]}")
    
    # Initialize uploader
    uploader = QdrantUploader(api_key, cluster_url)
    
    # Upload embeddings
    success = uploader.upload_embeddings()
    
    if success:
        print("\n" + "=" * 40)
        print("Upload completed successfully!")
        
        # Verify upload
        print("\nVerifying upload...")
        uploader.verify_upload()
        
        print(f"\nYour embeddings are now available in Qdrant!")
        print(f"Collection: documents")
        print(f"Vector dimensions: 384")
        print(f"Distance metric: Cosine")
        
    else:
        print("\n" + "=" * 40)
        print("Upload failed!")
        print("Please check your API key and cluster URL.")

if __name__ == "__main__":
    main()
