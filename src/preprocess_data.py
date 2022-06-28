import argparse
import os
import pickle

import pandas as pd
from sklearn.feature_extraction import DictVectorizer


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 68)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df



def dump_pickle(obj, filename):
    with open(filename, "wb") as fp:
        return pickle.dump(obj, fp)



def preprocess(df: pd.DataFrame, dv: DictVectorizer, fit_dv: bool=False):
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical = ['PU_DO']
    numerical = ['trip_distance']
    dicts = df[categorical + numerical].to_dict(orient='records')

    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, dv


def run(raw_data_path: str, dest_path: str, dataset: str="green"):
    df_train = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-01.parquet"))

    df_valid = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-02.parquet"))

    df_test = read_dataframe(
        os.path.join(raw_data_path, f"{dataset}_tripdata_2022-03.parquet"))

    target = 'duration'
    y_train = df_train[target].values
    y_valid = df_train[target].values
    y_test = df_train[target].values

    dv = DictVectorizer()
    X_train, dv = preprocess(df_train, dv, fit_dv=True)
    X_valid, _ = preprocess(df_valid, dv, fit_dv=False)
    X_test, _ = preprocess(df_test, dv, fit_dv=False)

    os.makedirs(dest_path, exist_ok=True)

    dump_pickle(dv, os.path.join(dest_path, "dv.pkl"))
    dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    dump_pickle((X_valid, y_valid), os.path.join(dest_path, "valid.pkl"))
    dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw_data_path",
        help="Location of your raw data which is either in .parquet format"
    )

    parser.add_argument(
        "--dest_path",
        help="where you want to save your processed data."
    )

    args = parser.parse_args()

    run(args.raw_data_path, args.dest_path)