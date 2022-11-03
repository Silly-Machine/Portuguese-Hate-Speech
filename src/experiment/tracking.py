# Unable warnings
import os
import warnings

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Requeiremnts
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import mlflow
import socket
import os


class experiment:
    def __init__(
        self, exp_name, model, model_name, host="localhost", port=7500, pipeline=True):
        # Checking server
        experiment.cnx(host, port)
        # Experiment atributes
        self.model = model
        self.model_name = model_name
        self.pipeline = pipeline
        # Server connection
        mlflow.set_tracking_uri(f"http://{host}:{port}")
        mlflow.set_experiment(exp_name)

    def run(self, X_train, y_train, X_test, y_test, predictions=False):
        print("[MLFLOW][EXECUTION] running experiment")
        total_data = len(X_train) + len(X_test)
        test_split = len(X_test) / total_data

        # Pipeline experiment
        if self.pipeline:
            mlflow.sklearn.autolog()
            with mlflow.start_run():
                # Fit
                self.model.fit(X_train, y_train)
                # Predict
                y_predict = self.model.predict(X_test)
                # Evaluate
                pre = precision_score(y_test, y_predict)
                acc = accuracy_score(y_test, y_predict)
                rec = recall_score(y_test, y_predict)
                auc = roc_auc_score(y_test, y_predict)
                f1 = f1_score(y_test, y_predict)
                # Save
                mlflow.log_params({"total_data": total_data, "test_split": test_split})
                mlflow.sklearn.log_model(self.model, self.model_name)
                mlflow.log_metric("precision", pre)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("auc", auc)
                mlflow.log_metric("f1", f1)
                # Return
                if predictions:
                    print("[MLFLOW][FINISHED] experiment executed successfully")
                    return y_predict
                else:
                    print("[MLFLOW] [FINISHED] experiment executed successfully")
                    print(
                        f"model:{self.model_name} - acc:{acc} - rec:{rec} - auc:{auc} - f1:{f1} \n"
                    )

    @staticmethod
    def cnx(host, port):
        check = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            check.connect((host, port))
            print("[MLFLOW] [START] server already running")
        except:
            print("[MLFLOW] [START] starting server")
            os.system(
                f"poetry run mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./artifacts -p {port} &"
            )
