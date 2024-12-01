from src.logger import logging
from src.exception_handling import CustomException
import os , sys
from pymongo import MongoClient
import pandas as pd
import pickle


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
            print("CSV uploaded successfully")

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
  
        

     