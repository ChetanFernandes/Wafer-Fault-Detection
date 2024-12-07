- Poject name - Faulty wafer detection 
- Probelm Statement - To detect whether provided wafer samples are "Good" or "Bad" based dn wafer data given. This is classifcation problem.
- Data shape - (100,592)
Steps follwed:-

1. Data Ingestion - collected the raw data and stored it in database Mongo DB.

2. EDA and Feature Engineering done -  Clean and transform raw data into a format suitable for model training.
        Key Components:
        Preprocessing Pipelines:
          - Handle missing data (Filled values using Median - as outlier where not treated).
        Visualization of Data:-
          - To gather insights. 
        Sampling - To balance the class as there was significant imbalance in class
          - Used ADASYN (Adaptive Synthetic Sampling) to balance the class as there was significant imbalance in   class (-1 and 1).
        Normalize or scale features (e.g.,Scaler).
          - Use Robust scaler to scale the data sets as the outliers where not treated.  Robust Scaler is less sensitive to outliers 
            because it  uses  the median and the interquartile range (IQR) for scaling.
        Feature Engineering:
          - Dimensionality reduction (e.g., PCA).
        Tools:
          - Pandas, NumPy, scikit-learn, imblearn, Matplotlib, Seaborn

3. Model Training - Purpose: Train the ML model using processed data.
        Algorithm Selection:
          - Classification.
          - models = { "LR" : LogisticRegressionCV(),
                            "SVC" : SVC(kernel='linear'),
                            "LSVC" : LinearSVC(),
                            "RFC" : RandomForestClassifier(),
                            "ABC" : AdaBoostClassifier(),
                            "GBC" : GradientBoostingClassifier(),
                            "DTC" : DecisionTreeClassifier(),
                            "GNB" : GaussianNB()
                            } 
        Model Development:
          - Frameworks: scikit-learn, KFold, 
        Hyperparameter Tuning:
          - Grid search
        Infrastructure:
          - Local machines

4. Model Evaluation - Assess model performance and select the best candidate.
        Evaluation Metrics:
          - Classification: Accuracy, precision, recall, F1 score, ROC-AUC.
        Cross-Validation:
          - k-fold cross-validation for robust evaluation.
        Tools:
          - ML libraries and custom scripts for visualization (e.g., Matplotlib, Seaborn)
        Outcome -  
          - Model with highest auc score and accuracy score was GradientBoostingClassifier() with value of 75 (both auc and accuracy score)
          - Applying PCA  - Model with highest auc score and accuracy score was 'Simple vector classifier' and  "GaussianNB" 
             with value of 87.5(both auc and accuracy score)

5. Model Deployment - Serve the trained model in production for real-time or batch predictions.
        Key Components:
        Model Serialization:
         - Formats: Pickle
        Versioning Tool:
         - Git Hub
        Serving Frameworks:
          - Flask
        Infrastructure:
          -  Docker  
          -  Docker image of application stored in AWS ECR (Elastic Container Registry)
          -  Application was deployed using AWS App Runner (PAAS)
Pipeline:-
+-------------------+        +---------------------+        +------------------------+
|                   |        |                     |        |                        |
|    Data Source     +------->+ Data Preprocessing+------->+       Model Training      |
| (Mongo DB) |               |  (ETL, Feature Eng.)|        |  (Frameworks, Tuning)  |
|                   |        |                     |        |                        |
+-------------------+        +---------------------+        +------------------------+
                                                                       |
                                                                       v
                                                            +------------------------+
                                                            |   Model Evaluation     |
                                                            | (Metrics, Validation)  |
                                                            +------------------------+
                                                                       |
                                                                       v
                                                            +------------------------+
                                                            |    Model Deployment    |
                                                            | (Serving Frameworks)   |
                                                            +------------------------+
                                                                      



6. Flow Diagram - [Architecture](image-1.png)










