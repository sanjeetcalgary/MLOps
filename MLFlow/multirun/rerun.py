import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import argparse
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import ElasticNet

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type= float, required=False, default=0.5)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()

# regression metrics
def modelscore(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual,pred))
    r2 = r2_score(actual,pred)
    return mae,rmse,r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # read the dataset
    df = pd.read_csv("red-wine-quality.csv")
    df.to_csv("red-wine-quality.csv",index=False)
    df.to_csv("data/data.csv")
    #split
    X = df.drop("quality",axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)
    X_train.to_csv("data/train.csv")
    X_test.to_csv("data/test.csv")

    alpha = args.alpha
    l1_ratio = args.l1_ratio

    # start tracking
    mlflow.set_tracking_uri(uri="")
    print("The set tracking uri is ", mlflow.get_tracking_uri())

    exp = mlflow.set_experiment(experiment_name="my_first_experiment_1")
    print("Name: {}".format(exp.name))
    print("Experiment_id: {}".format(exp.experiment_id))
    print("Artifact Location: {}".format(exp.artifact_location))
    print("Tags: {}".format(exp.tags))
    print("Lifecycle_stage: {}".format(exp.lifecycle_stage))
    print("Creation timestamp: {}".format(exp.creation_time))

    # start run
    mlflow.start_run(run_name="run_1.0")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC1",
        "release.version": "2.0"
    }
    mlflow.set_tags(tags)
    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # model code
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    (mae,rmse,r2) = modelscore(y_test,prediction)

    params = {
        "alpha": alpha,
        "l1_ratio": l1_ratio
    }
    mlflow.log_params(params)

    metrics = {
        "mae" : mae,
        "rmse" : rmse,
        "r2-score" : r2
    }
    mlflow.log_metrics(metrics)

    # log model
    mlflow.sklearn.log_model(model, "runmodel")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

################## second run########################
# start run
    mlflow.start_run(run_name="run_1.1")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC2",
        "release.version": "2.1"
    }
    mlflow.set_tags(tags)
    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # model code
    model = ElasticNet(alpha=0.9, l1_ratio=0.9, random_state=42)
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    (mae,rmse,r2) = modelscore(y_test,prediction)

    params = {
        "alpha": 0.9,
        "l1_ratio": 0.9
    }
    mlflow.log_params(params)

    metrics = {
        "mae" : mae,
        "rmse" : rmse,
        "r2-score" : r2
    }
    mlflow.log_metrics(metrics)

    # log model
    mlflow.sklearn.log_model(model, "runmodel_1")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()

####################### Third run ####################
    # start run
    mlflow.start_run(run_name="run_1.2")
    tags = {
        "engineering": "ML platform",
        "release.candidate": "RC3",
        "release.version": "2.2"
    }
    mlflow.set_tags(tags)
    current_run = mlflow.active_run()
    print("Active run id is {}".format(current_run.info.run_id))
    print("Active run name is {}".format(current_run.info.run_name))

    # model code
    model = ElasticNet(alpha=0.4, l1_ratio=0.4, random_state=42)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    (mae, rmse, r2) = modelscore(y_test, prediction)

    params = {
        "alpha": 0.4,
        "l1_ratio": 0.4
    }
    mlflow.log_params(params)

    metrics = {
        "mae": mae,
        "rmse": rmse,
        "r2-score": r2
    }
    mlflow.log_metrics(metrics)

    # log model
    mlflow.sklearn.log_model(model, "runmodel_2")
    mlflow.log_artifacts("data/")
    artifacts_uri = mlflow.get_artifact_uri()
    print("The artifact path is", artifacts_uri)
    mlflow.end_run()
    run = mlflow.last_active_run()
    print("Recent Active run id is {}".format(run.info.run_id))
    print("Recent Active run name is {}".format(run.info.run_name))