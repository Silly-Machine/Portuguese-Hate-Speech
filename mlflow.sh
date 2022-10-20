echo 'starting MLflow local server ... \n'
poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts -p 7500