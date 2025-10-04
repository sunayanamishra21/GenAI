@echo off
echo Starting RAG Vector Database Query Interface...
echo.
echo The app will open in your default web browser.
echo If it doesn't open automatically, go to: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server when you're done.
echo.
echo Starting server...
python -m streamlit run streamlit_app.py --server.port 8501
pause
