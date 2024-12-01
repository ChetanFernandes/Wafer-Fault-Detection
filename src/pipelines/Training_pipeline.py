from src.logger import logging
from src.exception_handling import CustomException
from src.constant.constants import MONGODB_URL, db_name,collection_name
from src.components.data_ingestion import DataIngestion
from src.components.data_ingestion import Upload_data_DB
from src.components.data_transformation import DataTransformation
import sys
import os

if __name__== '__main__':
    #obj = Upload_data_DB()
    #obj.upload_data()

    data_ingestion_obj = DataIngestion()
    train_data_path,test_data_path, raw_data_path,x_data_path,y_data_path = data_ingestion_obj.export_colleection_as_dataframe(MONGODB_URL,db_name,collection_name)
    print(train_data_path,test_data_path,raw_data_path)

    #Creating instance of class
    data_transformation_obj = DataTransformation()

    #Call Functions of class using object of class
    preprocessor_file_path, processor = data_transformation_obj.get_data_transformation_obj(x_data_path)
    print(preprocessor_file_path)

    
    x_resampled, y_resampled = data_transformation_obj.class_Balancing(x_data_path,y_data_path)

    X_train,X_test,y_train,y_test = data_transformation_obj.train_test_split_data_transformation(x_resampled,y_resampled,processor)
    print(f"X_train shape: {X_train.shape} \nX_test shape: {X_test.shape}, \ny_test shape: {y_test.shape},\ny_train shape: {y_train.shape}")