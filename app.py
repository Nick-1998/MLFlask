import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open(X_test,'model.pkl', 'rb'))



if __name__ == '__main__':
    app.run()
