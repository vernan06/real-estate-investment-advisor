"""Repository-root entry point for Streamlit Community Cloud."""

from runpy import run_path
from pathlib import Path


APP_PATH = Path(__file__).resolve().parent / "streamlit_app" / "app.py"
run_path(str(APP_PATH), run_name="__main__")
