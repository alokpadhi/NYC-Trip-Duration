from audioop import add
from pathlib import Path
from typing import Tuple
import pandas as pd
import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error

import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

import mlflow

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner


@task
def read_dataframe(filename:Path) -> pd.DataFrame:
    """Read the dataframe and process it for the feature and taregets.

    Args:
        filename (Path): Training and validation data file path

    Returns:
        pd.DataFrame: Processed dataframe.
    """
    df = pd.read_parquet(filename)

    df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
    df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 68)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


@task
def add_features(df_train: pd.DataFrame,
                df_val: pd.DataFrame):
    """Use DictVectorizer() to transform the featuers and return the splited arrays.

    Args:
        df_train (pd.DataFrame): Training dataframe
        df_val (pd.DataFrame): Validation Dataframe
    """

    df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
    df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_train['DOLocationID']

    categorical = ['PU_DO']
    numerical = ['trip_distance']

    dv = DictVectorizer()

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    X_train = dv.fit_transform(train_dicts)

    val_dicts = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)

    target = 'duration'

    y_train = df_train[target].values
    y_val = df_val[target].values

    return X_train, X_val, y_train, y_val, dv


@task
def train_model_search(train, valid, y_val):
    def objective(params):
        with mlflow.start_run():
            mlflow.set_tag("model", "xgboost")
            mlflow.log_params(params)
            booster = xgb.train(
                params=params,
                dtrain=train,
                num_boost_round=1000,
                evals=[(valid, 'validation')],
                early_stopping_rounds=50
            )
            y_pred = booster.predict(valid)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
        
        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
        'learning_rate': hp.loguniform('learning_rate', -3, 0),
        'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
        'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
        'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
        'objective': 'reg:linear',
        'seed': 42
    }

    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=1,
        trials=Trials()
    )
    return best_result


@task
def train_best_model(train, valid, y_val, dv):
    with mlflow.start_run():
        best_params = {
        'learning_rate':0.15615748360138743,
        'max_depth':4,
        'min_child_weight':2.350889641981104,
        'objective': 'reg:linear',
        'reg_alpha':0.10619618494552516,
        'reg_lambda':0.028152089722374665,
        'seed':42
        }

        mlflow.log_params(best_params)

        booster = xgb.train(
            params=best_params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )

        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

        with open("models/preprocessor.b", "wb") as fp:
            pickle.dump(dv, fp)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")

        mlflow.xgboost.log_model(booster, artifact_path="models_mlflow")


@flow(task_runner=SequentialTaskRunner())
def main(train_path: Path=Path("/disk/nyc_taxi/dataset/green_tripdata_2022-01.parquet"),
        val_path: Path=Path("/disk/nyc_taxi/dataset/green_tripdata_2022-02.parquet")):
    
    mlflow.set_tracking_uri("http://ec2-18-116-100-14.us-east-2.compute.amazonaws.com:5000")
    mlflow.set_experiment("nyc-taxi-experiment")

    df_train = read_dataframe(train_path)
    df_val = read_dataframe(val_path)

    X_train, X_val, y_train, y_val, dv = add_features(df_train, df_val).result()
    
    train = xgb.DMatrix(X_train, label=y_train)
    valid = xgb.DMatrix(X_val, label=y_val)
    
    train_model_search(train, valid, y_val)
    train_best_model(train, valid, y_val, dv)


# main()