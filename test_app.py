#!/usr/bin/env python3
"""
Test script to verify the Streamlit app is working
"""

import requests
import time

def test_streamlit_app():
    """Test if the Streamlit app is accessible"""
    
    url = "http://localhost:8501"
    
    print("[TEST] Testing Streamlit app...")
    print(f"[URL] {url}")
    
    try:
        # Try to connect to the app
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            print("[SUCCESS] Streamlit app is running and accessible!")
            print(f"[STATUS] Code: {response.status_code}")
            print(f"[CONTENT] Type: {response.headers.get('content-type', 'Unknown')}")
            print(f"[SIZE] Response: {len(response.content)} bytes")
            
            # Check if it's the Streamlit app
            if "streamlit" in response.text.lower() or "rag" in response.text.lower():
                print("[CONTENT] Streamlit app content detected!")
            else:
                print("[WARNING] App is running but content might not be loading properly")
                
        else:
            print(f"[ERROR] App returned status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("[ERROR] Cannot connect to the app. Is it running?")
        print("[TIP] Try running: python -m streamlit run streamlit_app.py")
        
    except requests.exceptions.Timeout:
        print("[ERROR] Connection timeout. The app might be starting up.")
        
    except Exception as e:
        print(f"[ERROR] Error testing app: {e}")

if __name__ == "__main__":
    test_streamlit_app()
