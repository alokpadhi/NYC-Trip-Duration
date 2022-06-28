import argparse
import os
import pickle

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

import numpy as np

from hyperopt import hp, space_eval
from hyperopt.pyll import scope

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from param_optimization import load_pickle


HPO_EXPERIMENT_NAME = "random-forest-optimization"
EXPERIMENT_NAME = "random-forest-models"

SPACE = {
    'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
    'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
    'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
    'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    'random_state': 42
}


def mlflow_setup(TRACKING_SERVER_HOST):
    mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
    mlflow.set_experiment("random-forest-experiments")
    mlflow.sklearn.autolog()


def load_picke(filename):
    with open(filename, "rb") as fp:
        return pickle.load(fp)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))

    with mlflow.start_run():
        params = space_eval(SPACE, params)
        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        valid_rmse = mean_squared_error(y_valid, rf.predict(X_valid), squared=False)
        mlflow.log_metric("valid_rmse", valid_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_metrics", test_rmse)


def run(data_path, log_top):
    client = MlflowClient()

    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.rmse ASC"]
    )

    for run in runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view=ViewType.ACTIVE_ONLY,
        max_results=log_top,
        order_by=["metrics.test_rmse ASC"]
    )
    
    best_run_id = best_run[0].info.run_id
    model_uri = f"runs:/{best_run_id}/model"

    # register the model
    mlflow.register_model(model_uri=model_uri, name="nyc-tax-regressor-model")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        default="./output",
        help="location of your saved preprocessed data"
    )

    parser.add_argument(
        "--top_n",
        default=5,
        help="n number of top models to show on mlflow"
    )

    parser.add_argument(
        "--TRACKING_SERVER_HOST",
        default="http://127.0.0.1:5000",
        help="Please give your mlflow tracking uri"
    )

    args = parser.parse_args()

    # set up mlflow variables
    mlflow_setup(args.TRACKING_SERVER_HOST)

    run(args.data_path, args.top_n)