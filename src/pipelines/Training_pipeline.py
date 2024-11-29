from src.logger import logging
from src.exception_handling import CustomException
from src.constant.constants import MONGODB_URL, db_name,collection_name
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import Upload_data_DB
import sys
import os

if __name__== '__main__':
    #obj = Upload_data_DB()
    #obj.upload_data()

    data_ingestion_obj = DataIngestion()
    train_data_path,test_data_path, raw_data_path = data_ingestion_obj.export_colleection_as_dataframe(MONGODB_URL,db_name,collection_name)
    print(train_data_path,test_data_path,raw_data_path)



