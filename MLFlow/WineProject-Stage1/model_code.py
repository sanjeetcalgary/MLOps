import numpy as np
import pandas as pd
import logging
import warnings
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import joblib
import mlflow
import mlflow.sklearn

# python model_code.py --alpha_Hyperparameter 0.7 --l1_ratio_hyperparameter 0.6 - running in cmd

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# retrieve arguments from command line
parser = argparse.ArgumentParser()
parser.add_argument("--alpha_Hyperparameter",type=float, required=False, default=0.5)
parser.add_argument("--l1_ratio_hyperparameter", type=float, required=False, default= 0.5)
args = parser.parse_args()

# regression metrics
def modelscore(actual, pred):
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual,pred))
    r2 = r2_score(actual,pred)
    return mae,rmse,r2

# model code
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # read the dataset
    df = pd.read_csv("red-wine-quality.csv")
    df.to_csv("red-wine-quality.csv",index=False)
    #split
    X = df.drop("quality",axis=1)
    y = df['quality']

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

    alpha = args.alpha_Hyperparameter
    l1_ratio = args.l1_ratio_hyperparameter

    # track the experiment
    exp = mlflow.set_experiment(experiment_name="experiment_1")

    with mlflow.start_run(experiment_id=exp.experiment_id):
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train,y_train)
        prediction = model.predict(X_test)
        (mae,rmse,r2) = modelscore(y_test,prediction)
        joblib.dump(model,"red_wine_model.joblib")
        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        # logging for this experiment [Parameters]
        mlflow.log_param("alpha-hyperparameter" , alpha)
        mlflow.log_param("l1_ratio-hyperparameter", l1_ratio)

        # logging for this experiment [Metrics]
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2-Score", r2)

        # logging artifacts i.e. path where serialized model will be saved as artifact
        mlflow.sklearn.log_model(model,"winemodel")


    #print(model.predict([[50,13,0.4,0.80,2.9,0.088,8,30,1,3.5,0.77,11]]))