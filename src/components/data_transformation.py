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
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import train_test_split

@dataclass
class datatransformation_config:
    pre_processor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformatiomn_config = datatransformation_config()

    def get_data_transformation_obj(self,x_data_path,y_data_path):
        try:
            logging.info("Data Transformation Initiated")

            x_df = pd.read_csv(x_data_path)
            y_df = pd.read_csv(y_data_path)

            logging.info(f"Read the csv file -> \n{x_df.shape, x_df.head()}, \n{y_df.shape, y_df.head()}")

            #Dropping "Unnamed" columns as its not necessary
            x_df.drop([x_df.columns[0]], axis = 1 , inplace = True)
            logging.info (f"Shape of x_df \n{x_df.shape} \n{x_df.head()}")

            logging.info(f"Drop null values greater than 35-> {(x_df.isnull().mean())*100}")
            cols = []
            for col in x_df.columns:
                if (x_df[col].isnull().sum()) > 35:
                    x_df.drop([col], axis = 1 , inplace = True)
                    cols.append(col)

            logging.info(f"Columns having null values greater than 35 droped-> {len(cols)} and shape of data frame {x_df.shape}")

            logging.info(f"Filling the null values")
            for col in x_df.columns:
                        x_df[col] = x_df[col].fillna(x_df[col].median())

            logging.info(f"Null values filled. Count of Null values -> {x_df.isna().sum().sum()}")     

            logging.info(f" Dropping columns having SD = 0")

            columns_Zero_STD = []
            for col in x_df.columns:
                if x_df[col].std() == 0:
                    x_df.drop([col], axis = 1 , inplace = True)
                    columns_Zero_STD.append(col)

            logging.info(f"Columns having SD = 0 dropped-> {len(columns_Zero_STD)} and shape of data frame {x_df.shape}")
    
            logging.info(f" Create a pipeline")

            pipeline = Pipeline(steps = [
                                 ('RobustScaler', RobustScaler())
                                            ])
        
            logging.info(f" Pipelline_obj-> \n{pipeline}")   

            logging.info(f"Preparing pipeline using columns transformer")

            preprocessor = ColumnTransformer([
                                            ('num_pipeline',pipeline,x_df.columns)
                                             ])
            
            logging.info(f"Pipeline completed-> \n{preprocessor}")                            
        
            save_object(
                         file_path=self.data_transformatiomn_config.pre_processor_obj_file_path,
                         obj = preprocessor
                       ) 

            return self.data_transformatiomn_config.pre_processor_obj_file_path, preprocessor,x_df,y_df

        except Exception as e:
            raise CustomException (e,sys)
        
    def train_test_split(self,x_df,y_df):

        try:

            logging.info(f"Applying train test and split")    
            X_train,X_test,y_train,y_test = train_test_split(x_df,y_df, test_size = 0.30, random_state = 1)
            logging.info(f"\nX_train shape: {X_train.shape} \nX_test shape:{X_test.shape}, \ny_train{y_train.shape},\ny_test {y_test.shape}")
          
            
            logging.info(f" Converting Y_train and Y_test to serire")
            y_train = y_train.values.flatten()
            y_test = y_test.values.flatten()

            return  X_train,X_test,y_train,y_test
    
        except Exception as e:
                raise CustomException(e,sys)


    def class_Balancing(self,X_train,y_train):

        try:
            logging.info("Inside function balancing)")
            logging.info(f"Read the data file -> \n{X_train.shape}")

            logging.info(f" Applying ADASYN technique as class is Imbalance. \n{((y_train == 1).sum()), ((y_train == -1).sum())}")
            adasys = ADASYN(sampling_strategy={-1: 100, 1: 60}, random_state=42,n_neighbors=3)
            X_train, y_train = adasys.fit_resample(X_train, y_train)
            logging.info(f" SMOTE techinque completed. \n{X_train.shape}, \n{(y_train == 1).sum(), (y_train == -1).sum()}")

            return X_train, y_train
        
        except Exception as e:
                raise CustomException(e,sys)

    def scaling(self,X_train,X_test,y_train,y_test,processor):

        try:

            logging.info(f" Scaling Initiated")
            X_train = pd.DataFrame(processor.fit_transform(X_train), columns = processor.get_feature_names_out())
            X_test = pd.DataFrame(processor.transform(X_test), columns = processor.get_feature_names_out())
         
            logging.info(f" Scaling completed. Data post scaling -> \n{X_train.head()}, \n{X_test.head()}")
            logging.info(f"Shape of Data post scaling -> \n{X_train.shape, X_test.shape, y_train.shape, y_test.shape}")
            

            return X_train,X_test,y_train,y_test
        
        except Exception as e:
            raise CustomException(e,sys)

    