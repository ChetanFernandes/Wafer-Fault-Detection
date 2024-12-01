from src.exception_handling import CustomException
from src.logger import logging
import sys, os
import pandas as pd
import numpy as np

from sklearn.metrics import  accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from src.utilis import model_training, Save_Model, read_yaml_file
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

class ModelTrainerConfig():
    def __init__(self):
        self.trained_model_file_path = os.path.join("artifacts","model.pkl")
        self.model_config_file_path= os.path.join('config','model.yaml')

class ModelTrainer():
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    logging.info("Initiating Model_training")

    def initiate_model_training(self,X_train,X_test,y_train,y_test):
        try:
            models_short_listed = { "LR" : LogisticRegressionCV(),
                                "SVC" : SVC(),
                                "LSVC" : LinearSVC(),
                                "RFC" : RandomForestClassifier(),
                                "GNB" : GaussianNB()
                                        }
            
            logging.info("Calling model_training function from utilis")
            report, model_list = model_training(X_train,X_test,y_train,y_test,models_short_listed)

          
            logging.info(f"Model with highest accuracy is -> {max(report)}. Model name is {model_list[report.index(max((report)))]}")

            best_model_name = model_list[report.index(max(report))]
            best_model_object = models_short_listed[best_model_name]

            return best_model_name, best_model_object
        
        except Exception as e:
            raise CustomException(e,sys)
    
    def fine_tuning_model(self,X_train,y_train,X_test,y_test):
            try:
                models_short_listed = { "LR" : LogisticRegressionCV(max_iter=10000),
                                "SVC" : SVC(),
                                "LSVC" : LinearSVC(max_iter = 10000),
                                "RFC" : RandomForestClassifier(),
                                "GNB" : GaussianNB()
                                        }

    # def fine_tuning_model(self,X_train,y_train,X_test,y_test, best_model_name,best_model_object,models_short_listed) -> object:
                logging.info("Inside fine_tune model function")
                model_list_fine = []
                report_fine = []
        
                for i in range((len(models_short_listed))-1):
                    model_name = list(models_short_listed.keys())[i]
                    model_object = list(models_short_listed.values())[i]
                    logging.info(f" {model_name,model_object}")
                    model_param_grid = read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][model_name]["search_param_grid"]
                    grid_search = GridSearchCV(model_object, param_grid = model_param_grid, cv=2, n_jobs=-1, verbose=1)
                    grid_search.fit(X_train, y_train)
                    finetuned_model = model_object.set_params(**grid_search.best_params_)
                    finetuned_model.fit(X_train,y_train)
                    y_pred = finetuned_model.predict(X_test)
                    model_score = accuracy_score(y_test,y_pred)
                    model_list_fine.append(model_name)
                    report_fine.append(model_score)

                    logging.info(f"Accuracy score of finetuned_model {model_name} is {model_score * 100}") 


                
                logging.info(f"Model with highest accuracy after finetune -> {max(report_fine)}. Model name is {model_list_fine[report_fine.index(max((report_fine)))]}")

                best_model_name = model_list_fine[report_fine.index(max(report_fine))]
                best_model_object = models_short_listed[best_model_name]
                finetuned_model_object = best_model_object.set_params(**grid_search.best_params_)
                    
                file_path = self.model_trainer_config.trained_model_file_path
            
                Save_Model(self.model_trainer_config.trained_model_file_path, finetuned_model_object)

                return file_path

            except Exception as e:
                raise CustomException(e,sys)

             

    





            
            
            























            


