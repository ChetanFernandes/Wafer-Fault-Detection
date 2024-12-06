from src.logger import logging
from src.exception_handling import CustomException
import os , sys
from pymongo import MongoClient
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, roc_auc_score
import yaml



def upload_data_db(url):
        logging.info("Uploading Data to MongoDB")
        try:
            client = MongoClient(url)
            db = client['wafer']
            collection = db['wafer_collection']
            df = pd.read_csv("notebooks\data\wafer.csv")
            data = df.to_dict('records')
            collection.insert_many(data)
            logging.info("CSV uploaded successfully")
            logging.info(f" DB created successfully. Existing DB's are - {client.list_database_names()}")
            all_documents = collection.find()
            return "Data successfully uploaded to MongoDB"

        except Exception as e:
            raise Exception(e,sys)
        
def save_object(file_path, obj):
     try:
          dir_name = os.path.dirname(file_path)
          os.makedirs(dir_name, exist_ok=True)
          with open(file_path,"wb") as file_obj:
               pickle.dump(obj,file_obj)
     
     except Exception as e:
          raise CustomException(e,sys)
     
def model_training(X_train,X_test,y_train,y_test,models):
     try:
          model_list = []
          report = []
          kf = KFold(n_splits = 5, shuffle=True, random_state=None) 
          for i in range(len(models)):
               model = (list(models.values())[i])
               scores = cross_val_score(model, X_train, y_train, cv = kf)
               logging.info(f" Scores ->, {scores * 100}")
               logging.info(f" Mean Score, {np.mean(scores)}")
               model.fit(X_train,y_train)
               y_pred = model.predict(X_test)
               logging.info(f" clasification report -> {classification_report(y_test,y_pred)}")
               fpr, tpr, thresholds = roc_curve(y_test,y_pred)
               auc_score = roc_auc_score(y_test,y_pred)
               model_score = accuracy_score(y_test,y_pred)

               logging.info(f" fpr {fpr}, tpr {tpr}, threshold {thresholds}")
               logging.info(f" AUC Score, {auc_score}")
               logging.info(f" Model Score , {model_score}")
               logging.info(f"Accuracy score of {model} is {auc_score * 100}")
               logging.info(f"{'*'*35}")
               
               report.append(auc_score*100)
               model_list.append(list(models.keys())[i])
          return report, model_list
       
     except Exception as e:
               raise CustomException(e,sys)
     
def Save_Model(file_path,best_model):
      try:
          dir_name = os.path.dirname(file_path)
          os.makedirs(dir_name, exist_ok = True)
          with open(file_path,"wb") as file_obj:
                    pickle.dump(best_model,file_obj)

      except Exception as e:
               raise CustomException(e,sys)
      
def read_yaml_file(filename: str) -> dict:
        try:
            with open(filename, "rb") as yaml_file:
                return yaml.safe_load(yaml_file)
            
        except Exception as e:
            raise CustomException(e, sys)
        
def model_load_object( filepath:str):
       try:
            
            with open(filepath, "rb") as model:
                return pickle.load(model)  
              
            
       except Exception as e:
            raise CustomException(e, sys)
       

       

        

     