import argparse
import os
import pickle

import mlflow
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from hyperopt.pyll import scope

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def mlflow_setup(TRACKING_SERVER_HOST):
    mlflow.set_tracking_uri(TRACKING_SERVER_HOST)
    mlflow.set_experiment("random-forest-experiments")


def load_pickle(filename):
    with open(filename, "rb") as fp:
        return pickle.load(fp)


def run(data_path, num_trials):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_valid, y_valid = load_pickle(os.path.join(data_path, "valid.pkl"))

    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "RandomForest")
            mlflow.log_params(params)

            rf = RandomForestRegressor()
            rf.fit(X_train, y_train)

            y_pred = rf.predict(X_valid, y_valid)
            rmse = mean_squared_error(y_valid, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
            return {"loss": rmse, "status": STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }

    rstate = np.random.default_rng(42)

    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trails=Trials(),
        rstate=rstate
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_path",
        default="./output",
        help="location where your processed data is saved"
    )

    parser.add_argument(
        "--max_evals",
        default="50",
        help="Number of parameter evaluations for the optimizer"
    )

    parser.add_argument(
        "--TRACKING_SERVER_HOST",
        default="http://127.0.0.1:5000",
        help="Please give your mlflow tracking uri"
    )

    args = parser.parse_args()
    
    # MLflow is being setup on AWS with postgresql database
    TRACKING_SERVER_HOST = args.TRACKING_SERVER_HOST
    mlflow_setup(TRACKING_SERVER_HOST)

    run(args.data_path, args.max_evals)