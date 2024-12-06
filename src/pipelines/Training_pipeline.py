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


class Initiate_training_pipeline:

    def upload_data_from_DB(self):
        try:
            #Calling Data upload function
            obj = Upload_data_DB()
            message = obj.upload_data()
            print(message)
        except Exception as e:
            raise CustomException(e,sys)

    
    def Data_ingestion(self):
        try:
            #Creating instance of class  for Data Ingestion
            data_ingestion_obj = DataIngestion()
            train_data_path,test_data_path, raw_data_path,x_data_path,y_data_path = data_ingestion_obj.export_colleection_as_dataframe(MONGODB_URL,db_name,collection_name)
            print(train_data_path,test_data_path,raw_data_path)

            return  train_data_path,test_data_path, raw_data_path,x_data_path,y_data_path
        
        except Exception as e:
            raise CustomException(e,sys)

    def Data_transformation(self,x_data_path,y_data_path):
        try:

            #Creating instance of class for Data Transformation
            data_transformation = DataTransformation()
            #Calling functions of class using object of class
            preprocessor_file_path, processor,x_df,y_df= data_transformation.get_data_transformation_obj(x_data_path,y_data_path)
            print(preprocessor_file_path)

            # Calling function to split the data to train test and split
            X_train,X_test,y_train,y_test = data_transformation.train_test_split(x_df,y_df)

            # Calling function to apply Smothing technique as class in imbalance
            X_train,y_train = data_transformation.class_Balancing(X_train,y_train)
            logging.info(f"X_train shape: {X_train.shape}, \nX_test shape: {X_test.shape}, \ny_test shape: {y_test.shape},\ny_train shape: {y_train.shape}")

            # Calling function to apply scaling technique
            X_train,X_test,y_train,y_test= data_transformation.scaling(X_train,X_test,y_train,y_test,processor)
            logging.info(f"X_train shape: {X_train.shape}, \nX_test shape: {X_test.shape}, \ny_test shape: {y_test.shape},\ny_train shape: {y_train.shape}")

            return X_train,X_test,y_train,y_test
              
        except Exception as e:
            raise CustomException(e,sys)

    def Model_Training(self,X_train,X_test,y_train,y_test):
        try:
            #Creating instance of class for Model Training
            model_trainer = ModelTrainer()
            
            # Calling function to initaite model training
            # best_model_name, best_model_object = model_trainer.initiate_model_training(X_train,X_test,y_train,y_test)
            file_path = model_trainer.initiate_model_training(X_train,X_test,y_train,y_test)

            # Calling function to fine_tune_model
            #file_path = model_trainer.fine_tuning_model(X_train,y_train,X_test,y_test,best_model_name, best_model_object)
            logging.info(f"{file_path}")
            
            return file_path

        except Exception as e:
            raise CustomException(e,sys)
    
    def run_pipeline(self):
        try:
            train_data_path,test_data_path, raw_data_path,x_data_path,y_data_path = self.Data_ingestion()
            X_train,X_test,y_train,y_test = self.Data_transformation(x_data_path,y_data_path)
            file_path = self.Model_Training(X_train,X_test,y_train,y_test)
            
        except Exception as e:
            raise CustomException(e,sys)