#!/usr/bin/env python3
"""
Launcher script for the RAG Vector Database Query Interface
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Streamlit app"""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    app_file = script_dir / "streamlit_app.py"
    
    # Check if the app file exists
    if not app_file.exists():
        print(f"Error: {app_file} not found!")
        sys.exit(1)
    
    # Check if .env file exists
    env_file = script_dir / ".env"
    if not env_file.exists():
        print("Warning: .env file not found. Make sure your environment variables are set.")
    
    print("🚀 Starting RAG Vector Database Query Interface...")
    print(f"📁 App directory: {script_dir}")
    print(f"📄 App file: {app_file}")
    print("🌐 The app will open in your default browser")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 60)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ], cwd=script_dir)
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped by user")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
