from src.exception_handling import CustomException
from src.logger import logging
from src.utilis import save_object
import  pandas as pd 
import numpy as np
import sys, os
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer # HAndling Missing values
from sklearn.preprocessing import RobustScaler # Beacause outliers is no treated. It uses median 
from sklearn.compose import ColumnTransformer
from imblearn.combine import SMOTETomek
from sklearn.model_selection import train_test_split

@dataclass
class datatransformation_config:
    pre_processor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformatiomn_config = datatransformation_config()

    def get_data_transformation_obj(self,x_data_path):
        try:
            logging.info("Data Transformation Initiated")

            x_df = pd.read_csv(x_data_path)
            logging.info(f"Read the csv file -> \n{x_df.head()}")

            logging.info("Selecting only numerical colummns")
            numerical_cols = []
            for col in x_df.columns:
                if x_df[col].dtypes != "O": 
                    numerical_cols.append(col)

            logging.info(f" Numerical columns -> \n{numerical_cols}")
  
            logging.info(f" Create a pipeline")

            pipeline = Pipeline(steps = [
                                 ('RobustScaler', RobustScaler())
                                            ])
            

            logging.info(f" Pipelline_obj-> \n{pipeline}")   


            logging.info(f"Preparing pipeline using columns transformer")


            preprocessor = ColumnTransformer([
                                            ('num_pipeline',pipeline,numerical_cols)
                                             ])
            
            logging.info(f"Pipeline completed-> \n{preprocessor}")                            
        
            save_object(
                         file_path=self.data_transformatiomn_config.pre_processor_obj_file_path,
                         obj = preprocessor
                       ) 

            return self.data_transformatiomn_config.pre_processor_obj_file_path, preprocessor

        except Exception as e:
            raise CustomException (e,sys)
        
    def class_Balancing(self,x_data_path, y_data_path):

        try:
            x_df = pd.read_csv(x_data_path)
            y_df = pd.read_csv(y_data_path)
    

            logging.info(f"Dropping column for X_df -> {x_df.columns[0]}")
            x_df.drop([x_df.columns[0]], axis = 1, inplace = True)
            logging.info(f" Coloumn dropped successfully -> \n{x_df.columns, x_df.shape, y_df.shape}")


            logging.info(f"Applying Inputer to get rid of null values -> {x_df.isna().sum().sum()}")
            imputer = KNNImputer(n_neighbors=3, weights='uniform')
            imputed_data = imputer.fit_transform(x_df)
            x_df = pd.DataFrame(imputed_data, columns=x_df.columns)
            logging.info(f"Null values filled -> {x_df.isna().sum().sum()}")


            logging.info(f" Applying SMOTE technique as Target variable in Imbalance. \n{y_df.value_counts()}")
            smote_tomek = SMOTETomek(random_state = 1)
            x_resampled,y_resampled = smote_tomek.fit_resample(x_df,y_df)
            logging.info(f" SMOTE techinque completed. \n{x_resampled.shape, y_resampled.shape,y_resampled.value_counts()}")

            return x_resampled, y_resampled
        
        except Exception as e:
                raise CustomException(e,sys)

    def train_test_split_data_transformation(self,x_resampled,y_resampled,processor):

        try:

            logging.info(f"Applying train test and split")    
            X_train,X_test,y_train,y_test = train_test_split(x_resampled,y_resampled, test_size = 0.30, random_state = 1)
            logging.info(f"\nX_train shape: {X_train.shape} \nX_test shape:{X_test.shape}, \ny_train{y_train.shape},\ny_test {y_test.shape}")

            #processor_obj = self.get_data_transformation_obj[1]
            #logging.info(f"Reading processor object from  -> {processor_obj}")


            logging.info(f" Data tranformation Initiated")
            X_train = pd.DataFrame(processor.fit_transform(X_train), columns = processor.get_feature_names_out())
            X_test = pd.DataFrame(processor.transform(X_test), columns = processor.get_feature_names_out())
         
            logging.info(f" Data tranformation completed. Data post transfoemation -> \n{X_train.head()}, \n{X_test.head()}")
            logging.info(f"Shape of Data post transformation -> \n{X_train.shape, X_test.shape, y_train.shape, y_test.shape}")
            
            logging.info(f" Converting Y_train and Y_test to serire")
            y_train = y_train.values.flatten()
            y_test = y_test.values.flatten()

            logging.info(f"{y_test}")
            logging.info(f"{y_train}")
            return X_train,X_test,y_train,y_test

    
        except Exception as e:
            raise CustomException(e,sys)

    