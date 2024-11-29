from src.logger import logging
from src.exception_handling import CustomException
import os, sys
import numpy
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utilis import upload_data_db
from src.constant import *
from pymongo import MongoClient
from sklearn.model_selection import train_test_split

class Upload_data_DB:
    def upload_data(self):
        try:
            url =  "mongodb+srv://chetan1:chetan1@cluster0.2c8ti.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
            upload_data_db(url)
        except Exception as e:
            raise Exception (e,sys)
    
@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts", "train_csv")
    test_data_path:str = os.path.join("artifacts", "test_csv")
    raw_data_path:str = os.path.join("artifacts", "raw_csv")
    x_data_path:str = os.path.join("artifacts","x_csv")
    y_data_path:str = os.path.join("artifacts","y_csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def export_colleection_as_dataframe(self, MONGODB_URL,db_name,collection_name):
        logging.info('Data Ingestion method starts')

        try:
            client = MongoClient(MONGODB_URL)
            db = client[db_name]
            collection = db[collection_name]
            df = pd.DataFrame(list(collection.find()))
            logging.info(f" \n{df.head()}")

            if "_id" in df.columns:
                df.drop(columns=["_id"], axis = 1 , inplace = True)
            df.replace({"na": np.nan}, inplace=True)

            logging.info (f" After removing column '_id'-> \n{df}")
            
            x = df.drop(columns = (list(df.columns))[-1], axis = 1)
            y = df["Good/Bad"]

            logging.info("Data set read as Pandas DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            train_set, test_set = train_test_split(df,test_size=0.30, random_state=1)

            df.to_csv(self.ingestion_config.raw_data_path, index = False)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            x.to_csv(self.ingestion_config.x_data_path, index = False, header = True)
            y.to_csv(self.ingestion_config.y_data_path, index = False, header = True)

            logging.info("Ingestion of data is completed")

            return(
                   self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path, 
                   self.ingestion_config.raw_data_path
                 )


        except Exception as e:
            raise Exception(e, sys)

