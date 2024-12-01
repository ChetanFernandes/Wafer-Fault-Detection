from src.logger import logging
from src.exception_handling import CustomException
from src.constant.constants import MONGODB_URL, db_name,collection_name
from src.components.data_ingestion import DataIngestion, Upload_data_DB
from src.components.data_transformation import DataTransformation
from src.components.Model_trainer import ModelTrainer
import sys
import os
import warnings
warnings.filterwarnings('ignore')


if __name__== '__main__':
    #obj = Upload_data_DB()
    #obj.upload_data()

    #Creating instance of class  for Data Ingestion
    data_ingestion_obj = DataIngestion()

    train_data_path,test_data_path, raw_data_path,x_data_path,y_data_path = data_ingestion_obj.export_colleection_as_dataframe(MONGODB_URL,db_name,collection_name)
    print(train_data_path,test_data_path,raw_data_path)

    #Creating instance of class for Data Transformation
    data_transformation_obj = DataTransformation()

    #Calling functions of class using object of class
    preprocessor_file_path, processor = data_transformation_obj.get_data_transformation_obj(x_data_path)
    print(preprocessor_file_path)

    # Calling function to balance class
    x_resampled, y_resampled = data_transformation_obj.class_Balancing(x_data_path,y_data_path)

    # Calling function to train_test_split
    X_train,X_test,y_train,y_test = data_transformation_obj.train_test_split_data_transformation(x_resampled,y_resampled,processor)
    logging.info(f"X_train shape: {X_train.shape}, \nX_test shape: {X_test.shape}, \ny_test shape: {y_test.shape},\ny_train shape: {y_train.shape}")


    #Creating instance of class for Model Training
    model_trainer = ModelTrainer()
    
    # Calling function to initaite model training
    best_model_name, best_model_object = model_trainer.initiate_model_training(X_train,X_test,y_train,y_test)

    # Calling function to fine_tune_model
    file_path = model_trainer.fine_tuning_model(X_train,y_train,X_test,y_test)
    print(file_path)

    