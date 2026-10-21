
from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle

# Flask app
app = Flask(__name__)



if __name__ == '__main__':
    app.run(debug=True)