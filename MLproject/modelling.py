import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

mlflow.set_tracking_uri("http://127.0.0.1:5000/")

mlflow.set_experiment("Students Habit Performance")

data = pd.read_csv('dataset_preprocessing.csv')

X = data.drop("exam_score", axis=1)
y = data["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    #Log Parameters
    n_estimators = 100
    max_depth = 5
    mlflow.autolog()

    #Train Model
    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path = "model",
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #Log Metrics
    r2_Score = r2_score(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred)
    
    mlflow.log_metric("r2_score", r2_Score)
    mlflow.log_metric("rmse", r2_Score)