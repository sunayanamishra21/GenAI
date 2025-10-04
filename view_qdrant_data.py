import json
import requests
import numpy as np
from typing import List, Dict, Optional
import uuid

class QdrantViewer:
    def __init__(self, api_key: str, cluster_url: str):
        self.api_key = api_key
        self.cluster_url = cluster_url.rstrip('/')
        self.collection_name = "documents"
        
    def get_collection_info(self) -> Dict:
        """Get information about the collection"""
        headers = {}
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            response = requests.get(
                f"{self.cluster_url}/collections/{self.collection_name}",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Failed to get collection info: {response.status_code} - {response.text}")
                return {}
                
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {}
    
    def get_points(self, limit: int = 10, offset: int = 0) -> List[Dict]:
        """Get points from the collection"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            response = requests.post(
                f"{self.cluster_url}/collections/{self.collection_name}/points/scroll",
                headers=headers,
                json={
                    "limit": limit,
                    "offset": offset,
                    "with_payload": True,
                    "with_vectors": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", {}).get("points", [])
            else:
                print(f"Failed to get points: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error getting points: {e}")
            return []
    
    def search_similar(self, query_vector: List[float], limit: int = 5) -> List[Dict]:
        """Search for similar vectors"""
        headers = {
            "Content-Type": "application/json"
        }
        
        if self.api_key:
            headers["api-key"] = self.api_key
        
        try:
            response = requests.post(
                f"{self.cluster_url}/collections/{self.collection_name}/points/search",
                headers=headers,
                json={
                    "vector": query_vector,
                    "limit": limit,
                    "with_payload": True,
                    "with_vectors": True
                },
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("result", [])
            else:
                print(f"Failed to search: {response.status_code} - {response.text}")
                return []
                
        except Exception as e:
            print(f"Error searching: {e}")
            return []

def display_collection_info(viewer: QdrantViewer):
    """Display collection information"""
    print("=" * 60)
    print("COLLECTION INFORMATION")
    print("=" * 60)
    
    info = viewer.get_collection_info()
    if info:
        result = info.get("result", {})
        print(f"Collection Name: {viewer.collection_name}")
        print(f"Points Count: {result.get('points_count', 'N/A')}")
        print(f"Status: {result.get('status', 'N/A')}")
        
        vectors_config = result.get('config', {}).get('params', {}).get('vectors', {})
        if isinstance(vectors_config, dict):
            print(f"Vector Size: {vectors_config.get('size', 'N/A')}")
            print(f"Distance Metric: {vectors_config.get('distance', 'N/A')}")
        else:
            print(f"Vector Config: {vectors_config}")
    else:
        print("Failed to retrieve collection information")
    
    print()

def display_points(viewer: QdrantViewer, limit: int = 5):
    """Display sample points with their data"""
    print("=" * 60)
    print(f"SAMPLE POINTS (showing {limit} points)")
    print("=" * 60)
    
    points = viewer.get_points(limit=limit)
    
    if not points:
        print("No points found or failed to retrieve points")
        return
    
    for i, point in enumerate(points, 1):
        print(f"\n--- POINT {i} ---")
        print(f"ID: {point.get('id', 'N/A')}")
        
        payload = point.get('payload', {})
        print(f"Source: {payload.get('source', 'N/A')}")
        print(f"Chunk ID: {payload.get('chunk_id', 'N/A')}")
        print(f"Original ID: {payload.get('original_id', 'N/A')}")
        
        # Display text chunk (first 200 characters)
        text = payload.get('text', '')
        if text:
            display_text = text[:200] + "..." if len(text) > 200 else text
            print(f"Text: {display_text}")
        
        # Display vector information
        vector = point.get('vector', [])
        if vector:
            print(f"Vector dimensions: {len(vector)}")
            print(f"Vector (first 10 values): {vector[:10]}")
            print(f"Vector (last 10 values): {vector[-10:]}")
            print(f"Vector norm: {np.linalg.norm(vector):.6f}")
        
        print("-" * 40)

def display_vector_statistics(viewer: QdrantViewer):
    """Display vector statistics"""
    print("=" * 60)
    print("VECTOR STATISTICS")
    print("=" * 60)
    
    points = viewer.get_points(limit=100)  # Get more points for statistics
    
    if not points:
        print("No points found for statistics")
        return
    
    vectors = []
    vector_norms = []
    
    for point in points:
        vector = point.get('vector', [])
        if vector:
            vectors.append(vector)
            vector_norms.append(np.linalg.norm(vector))
    
    if vectors:
        vectors_array = np.array(vectors)
        
        print(f"Sample size: {len(vectors)} vectors")
        print(f"Vector dimensions: {len(vectors[0])}")
        print(f"Mean norm: {np.mean(vector_norms):.6f}")
        print(f"Std norm: {np.std(vector_norms):.6f}")
        print(f"Min norm: {np.min(vector_norms):.6f}")
        print(f"Max norm: {np.max(vector_norms):.6f}")
        
        # Statistics for each dimension
        print(f"\nDimension-wise statistics:")
        print(f"Mean per dimension: {np.mean(vectors_array, axis=0)[:5]}... (showing first 5)")
        print(f"Std per dimension: {np.std(vectors_array, axis=0)[:5]}... (showing first 5)")

def interactive_search(viewer: QdrantViewer):
    """Interactive search functionality"""
    print("=" * 60)
    print("INTERACTIVE SEARCH")
    print("=" * 60)
    
    # Get a sample vector to search with
    points = viewer.get_points(limit=1)
    if not points:
        print("No points available for search")
        return
    
    sample_vector = points[0].get('vector', [])
    if not sample_vector:
        print("No vector available for search")
        return
    
    print("Searching for similar vectors using a sample vector...")
    print(f"Using vector from point: {points[0].get('id', 'N/A')}")
    
    similar_points = viewer.search_similar(sample_vector, limit=3)
    
    if similar_points:
        print(f"\nFound {len(similar_points)} similar points:")
        
        for i, point in enumerate(similar_points, 1):
            print(f"\n--- SIMILAR POINT {i} ---")
            print(f"ID: {point.get('id', 'N/A')}")
            print(f"Similarity Score: {point.get('score', 'N/A')}")
            
            payload = point.get('payload', {})
            text = payload.get('text', '')
            if text:
                display_text = text[:150] + "..." if len(text) > 150 else text
                print(f"Text: {display_text}")
    else:
        print("No similar points found")

def main():
    # Your configuration
    api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xFcDgqa1ADs22JIIwFNvrN45FWcgRQQgqZ1AetfKoSU"
    cluster_url = "https://02ac6bb1-cdd6-4187-8bdc-f7f82d20ff0d.us-east4-0.gcp.cloud.qdrant.io"
    
    print("Qdrant Vector Database Viewer")
    print("=" * 60)
    print(f"Cluster URL: {cluster_url}")
    print(f"Collection: documents")
    print(f"API Key: {'*' * 20}...{api_key[-10:]}")
    print()
    
    # Initialize viewer
    viewer = QdrantViewer(api_key, cluster_url)
    
    # Display collection information
    display_collection_info(viewer)
    
    # Display sample points
    display_points(viewer, limit=3)
    
    # Display vector statistics
    display_vector_statistics(viewer)
    
    # Interactive search
    interactive_search(viewer)
    
    print("\n" + "=" * 60)
    print("VIEWING COMPLETED")
    print("=" * 60)
    
    # Menu for additional operations
    print("\nAdditional Options:")
    print("1. View more points")
    print("2. View specific point by ID")
    print("3. Search with custom query")
    print("4. Export data to files")
    
    choice = input("\nEnter your choice (1-4) or press Enter to exit: ").strip()
    
    if choice == "1":
        limit = int(input("Enter number of points to view (default 10): ") or "10")
        display_points(viewer, limit=limit)
        
    elif choice == "2":
        point_id = input("Enter point ID to view: ").strip()
        if point_id:
            # Get specific point by ID
            headers = {}
            if api_key:
                headers["api-key"] = api_key
            
            try:
                response = requests.post(
                    f"{cluster_url}/collections/documents/points/scroll",
                    headers=headers,
                    json={
                        "filter": {
                            "must": [
                                {
                                    "key": "id",
                                    "match": {"value": point_id}
                                }
                            ]
                        },
                        "with_payload": True,
                        "with_vectors": True
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    points = result.get("result", {}).get("points", [])
                    if points:
                        display_points(QdrantViewer(api_key, cluster_url), limit=1)
                    else:
                        print("Point not found")
                else:
                    print(f"Failed to retrieve point: {response.status_code} - {response.text}")
                    
            except Exception as e:
                print(f"Error retrieving point: {e}")
    
    elif choice == "3":
        print("Custom search functionality would require a query embedding.")
        print("You can use the sentence-transformers library to create embeddings for your query.")
        
    elif choice == "4":
        print("Exporting data to files...")
        points = viewer.get_points(limit=1000)  # Get more points for export
        
        if points:
            # Export to JSON
            with open("qdrant_export.json", "w", encoding="utf-8") as f:
                json.dump(points, f, indent=2, ensure_ascii=False)
            
            # Export text chunks
            with open("qdrant_text_chunks.txt", "w", encoding="utf-8") as f:
                for i, point in enumerate(points, 1):
                    payload = point.get("payload", {})
                    text = payload.get("text", "")
                    f.write(f"=== CHUNK {i} (ID: {point.get('id', 'N/A')}) ===\n")
                    f.write(text)
                    f.write("\n\n")
            
            print("Data exported to:")
            print("- qdrant_export.json (complete data)")
            print("- qdrant_text_chunks.txt (text only)")

if __name__ == "__main__":
    main()
