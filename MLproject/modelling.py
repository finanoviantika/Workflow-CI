import mlflow
import pandas as pd
import numpy as np
import os
import warnings
import sys
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    file_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_preprocessing.csv")
    data = pd.read_csv(file_path)
    
    X = data.drop("exam_score", axis=1)
    y = data["exam_score"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    #Log Parameters
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    
    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path = "model",
        )
        
        r2_Score = r2_score(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        maxerr = max_error(y_test, y_pred)
        
        mlflow.log_metric("r2_score", r2_Score)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("max_error", maxerr)
