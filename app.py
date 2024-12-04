from flask import render_template, jsonify, Flask, request, send_file
from src.exception_handling import CustomException
from src.logger import logging
import sys, os
from src.pipelines.Training_pipeline import Initiate_training_pipeline

app = Flask(__name__)

@app.route("/")
def route():
    return "Welcome to my application"

@app.route("/train")
def train_route():
    try:
        if request.method == 'GET':
            train_pipeline = Initiate_training_pipeline()
            train_pipeline.run_pipeline()
            return "Training Completed"
        
        else:
            return  "Incorrect Method"
        
    except Exception as e:
        raise Exception(e,sys)

if __name__ == "__main__
    app.run(host="0.0.0.0", port=5000, debug= True)
