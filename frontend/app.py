from flask import Flask, render_template
import torch
import numpy as np


app = Flask(__name__)

@app.route('/')
@app.route("/home")

def home():
    return render_template('index.html')


# @app.route('/', methods=['POST'])

# def predict():

#     return render_template('index.html', prediction=classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)

