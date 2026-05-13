#!/usr/bin/env python
"""
Run the Visualization GUI.

Usage:
    python run_gui.py

Or make executable and run:
    chmod +x run_gui.py
    ./run_gui.py
"""

import os
import sys

# Get project root (parent of gui/)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

# Ensure we're using the virtual environment if available
venv_python = os.path.join(project_root, '.venv', 'bin', 'python')

if os.path.exists(venv_python) and sys.executable != venv_python:
    # Re-run with venv python
    os.execl(venv_python, venv_python, *sys.argv)

# Add gui folder to path for import
sys.path.insert(0, script_dir)

# Run the GUI
from gui import main

if __name__ == "__main__":
    main()

