import argparse
import os
import pickle

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import mlflow


def mlflow_setup(TRACKING_SERVER_HOST):
    mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
    mlflow.set_experiment("nyc-taxi-experiments")


def load_pickle(filename:str):
    with open(filename, "rb") as fp:
        return pickle.load(fp)


def run(data_path):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        mlflow.set_tag("datascientist", "alok")
        mlflow.sklearn.autolog()
        
        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_valid)

        rmse = mean_squared_error(y_valid, y_pred, squared=False)

        mlflow.log_metric("rmse", rmse)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="./output",
        help="location of your saved preprocessed data"
    )

    args = parser.parse_args()

    run(args.data_path)