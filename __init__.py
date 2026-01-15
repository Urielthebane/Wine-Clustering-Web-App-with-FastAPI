# app/__init__.py

from flask import Flask

# Create Flask app instance
app = Flask(__name__)

# Import routes after app is created
from app import main
