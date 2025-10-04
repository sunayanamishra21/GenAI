@echo off
echo Starting RAG Vector Database Query Interface...
echo.
echo This will launch the Streamlit web application in your browser.
echo Press Ctrl+C to stop the server when you're done.
echo.
pause
python -m streamlit run streamlit_app.py --server.port 8501
pause
