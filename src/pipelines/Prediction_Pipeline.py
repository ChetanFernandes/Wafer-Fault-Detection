from src.exception_handling import CustomException
from src.logger import logging
import sys, os
from dataclasses import dataclass
#from flask import request
import pandas as pd
import numpy as np
from src.utilis import model_load_object
from src.components.data_transformation import DataTransformation


@dataclass
class Prediction_pipeline_config:
    prediction_output_dirname: str = "predictions"
    prediction_file_name:str =  "predicted_file.csv"
    predicted_file_path:str = os.path.join(prediction_output_dirname,prediction_file_name)


class prediction_pipe_line:
    
    def __init__(self, request):
       self.request = request
       self.prediction_pipeline_config = Prediction_pipeline_config()

    
    def save_input_file(self):
        try:
            self.pred_file_input_dir = "prediction_artifacts"
            os.makedirs(self.pred_file_input_dir, exist_ok=True)
            Input_csv_file = self.request.files['file']
            pred_file_path = os.path.join(self.pred_file_input_dir, Input_csv_file.filename)
            print(pred_file_path)
            Input_csv_file.save(pred_file_path)
            return pred_file_path
    
        except Exception as e:
            raise Exception(e,sys)
    
        
    def cleaning_the_prediction_file(self,pred_file_path):
        try:
            logging.info("reading the dataframe ")
            df_pred:pd.DataFrane = pd.read_csv(pred_file_path)
            logging.info(f"{df_pred.shape}")

            df_pred.drop("Unnamed: 0" , axis = 1 , inplace = True) if ("Unnamed: 0") in df_pred.columns else df_pred
            df_pred.drop("Good/Bad" , axis = 1 , inplace = True) if ("Good/Bad") in df_pred.columns else df_pred
            logging.info (f"Shape of data post dropping columns -> {df_pred.shape}")

          
            logging.info(f" Post dropping of columns -> {df_pred.shape}")

            logging.info(f"Check for null values - {df_pred.isna().sum().sum()}")
            
            logging.info(f"Filling the null values")
            for col in list(df_pred.columns):
                        df_pred[col] = df_pred[col].fillna(df_pred[col].median())

            logging.info(f"Null values filled. Count of Null values -> {df_pred.isna().sum().sum()}")     
           
            logging.info(f"{self.pred_file_input_dir}")
            os.makedirs(self.pred_file_input_dir, exist_ok=True)

            df_pred.to_csv(os.path.join(self.pred_file_input_dir, "Cleaned.csv"), index = False)
            
            cleaned_file_path = os.path.join(self.pred_file_input_dir, "Cleaned.csv")
            return cleaned_file_path

        except Exception as e:
            raise Exception(e,sys)
        
    def predict(self,df_pred):

        try:
            model_file_path:str = os.path.join("artifacts",'model.pkl')
            processor_file_path:str = os.path.join("artifacts","preprocessor.pkl")
            
            model = model_load_object(model_file_path)
            processor = model_load_object(processor_file_path)
            
            transformed_x = processor.transform(df_pred)

            preds = model.predict(transformed_x)
            
            return preds
        
        except Exception as e:
            raise Exception(e,sys)
        
    def predicted_file_cleaning(self, cleaned_file_path):
        try:
            df_pred:pd.DataFrane = pd.read_csv(cleaned_file_path)
            prediction_column_name = "Result"

            predictions = self.predict(df_pred)

            df_pred[prediction_column_name] = [pred for pred in predictions]
            target_column_mapping = {-1:'bad', 1:'good'}

            df_pred[prediction_column_name] = df_pred[prediction_column_name].map(target_column_mapping)

            os.makedirs(os.path.dirname(self.prediction_pipeline_config.predicted_file_path), exist_ok=True)

            df_pred.to_csv(self.prediction_pipeline_config.predicted_file_path, index= False)
            logging.info(f"predictions completed {self.prediction_pipeline_config.predicted_file_path}")
      
        except Exception as e:
            raise Exception(e,sys)
            

    def run_pipeline(self):
        try:

            pred_file_path = self.save_input_file()
            cleaned_file_path = self.cleaning_the_prediction_file(pred_file_path)
            self.predicted_file_cleaning(cleaned_file_path)
            return  self.prediction_pipeline_config
     
    
        except Exception as e:
            raise Exception(e,sys)

    




