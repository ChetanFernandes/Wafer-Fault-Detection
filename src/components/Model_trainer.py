from src.exception_handling import CustomException
from src.logger import logging
from src.utilis import model_training,read_yaml_file,Save_Model
import sys, os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
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
            models = { "LR" : LogisticRegression(),
                        "SVC" : SVC(),
                        "LSVC" : LinearSVC(),
                        "RFC" : RandomForestClassifier(),
                        "ABC" : AdaBoostClassifier(),
                        "GBC" : GradientBoostingClassifier(),
                        "DTC" : DecisionTreeClassifier(),
                        "GNB" : GaussianNB(),
                        "BRFC": BalancedRandomForestClassifier()
                        }
            
            logging.info("Calling model_training function from utilis")
            report, model_list = model_training(X_train,X_test,y_train,y_test,models)

            logging.info(f"Model with highest auc score is -> {max(report)}. Model name is {model_list[report.index(max((report)))]}")

            best_model_name = model_list[report.index(max(report))]
            best_model_object = models[best_model_name]

            file_path = self.model_trainer_config.trained_model_file_path

            Save_Model(self.model_trainer_config.trained_model_file_path, best_model_object)

            return file_path

        
        except Exception as e:
            raise CustomException(e,sys)
    
    def fine_tuning_model(self,X_train,y_train,X_test,y_test,best_model_name, best_model_object):
            logging.info("Inside fine_tune model function")
            try:
            
                model_param_grid = read_yaml_file(self.model_trainer_config.model_config_file_path)["model_selection"]["model"][best_model_name]["search_param_grid"]
                grid_search = GridSearchCV(best_model_object, param_grid = model_param_grid, cv=2, n_jobs=1, verbose=5)
                grid_search.fit(X_train, y_train)
                finetuned_model = best_model_object.set_params(**grid_search.best_params_)
                y_pred = finetuned_model.predict(X_test)
                auc_score = roc_auc_score(y_test,y_pred)
                model_score = accuracy_score(y_test,y_pred)

                 
                logging.info(f"Best parameters of finetuned model of  {grid_search.best_params_}, {finetuned_model}")

                logging.info(f"Accuracy score of finetuned_model {best_model_name} is {model_score * 100}")
        
                logging.info(f"AUC score of finetuned_model {best_model_name} is {auc_score * 100}") 

                finetuned_model_object = best_model_object.set_params(**grid_search.best_params_)
                    
                file_path = self.model_trainer_config.trained_model_file_path
            
                Save_Model(self.model_trainer_config.trained_model_file_path, finetuned_model_object)

                return file_path

            except Exception as e:
                raise CustomException(e,sys)

             

    





            
            
            























            


