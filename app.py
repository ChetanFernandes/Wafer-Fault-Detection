from flask import render_template, jsonify, Flask, request, send_file
from src.exception_handling import CustomException
from src.logger import logging
import sys, os
from src.pipelines.Training_pipeline import Initiate_training_pipeline
from src.pipelines.Prediction_Pipeline import prediction_pipe_line


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

    
@app.route("/predict", methods = ["GET","POST"])
def upload():
    try:
        if request.method == "POST":

            upload1 = prediction_pipe_line(request)

            prediction_pipeline_config = upload1.run_pipeline()
            return send_file(prediction_pipeline_config.predicted_file_path,
                            download_name = prediction_pipeline_config.prediction_file_name,
                            as_attachment = True)


        else:
            return render_template('upload_file.html')
    
    except Exception as e:
        raise Exception(e,sys)
    

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug= True)
