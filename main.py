from datetime import datetime
import joblib

from dagster import (
    job, op,
    Out, Output,
    AssetMaterialization,
    get_dagster_logger
)

from pandas import DataFrame

from modules.get_data import fetch_data_
from modules.setup_folders import setup_folders_
from modules.model_training import model_training_
from modules.data_validation import validate_data_
from modules.model_evaluation import model_evaluation_
from modules.train_test_split import train_test_split_
from modules.test_split_transformation import test_split_transformation_
from modules.train_split_transformation import train_split_transformation_


@op
def setup_folders():
    setup_folders_()


@op
def download_data(setup):
    logger = get_dagster_logger()
    try:
        df = fetch_data_()
        logger.info("Download Successful!")
    except Exception as _:
        logger.error("Download Unsuccessful!")
        raise Exception("Data couldn't be downloaded.")

    return df


@op
def data_validation(context, df):
    logger = get_dagster_logger()

    result = validate_data_(df)

    df_clean = result["df_clean"]
    df_error_no = result["df_error_no"]

    # If more than half the rows are bad
    # Terminate run
    if df_error_no > len(df) * 0.5:
        logger.error("Too many bad rows!")
        raise Exception("Too many bad rows!")

    # Save the Clean Dataset
    try:
        df_clean.to_csv(
            f"./data/csv/{datetime.timestamp(datetime.now())}.csv"
        )

        context.log_event(
            AssetMaterialization(
                asset_key="df_clean",
                description="Persisted df_clean to storage."
            )
        )

        logger.info("Data Saved!")
    except Exception as _:
        logger.error("Couldn't Save Data!")
        raise Exception("Data couldn't be saved.")

    return df_clean


@op(out={
    "train_split": Out(dagster_type=DataFrame),
    "test_split": Out(dagster_type=DataFrame)
})
def train_test_split(df):
    result = train_test_split_(df)

    yield Output(
        value=result["train_split"],
        output_name="train_split"
    )

    yield Output(
        value=result["test_split"],
        output_name="test_split"
    )


@op(out={
    "targets": Out(),
    "features": Out(),
    "status_le": Out(),
    "age_le": Out(),
    "f_scaler": Out()
})
def train_split_transformation(context, df):
    now = datetime.now()
    timestamp = now.timestamp()
    logger = get_dagster_logger()

    result = train_split_transformation_(df)

    status_le = result["Status LE"]
    age_le = result["Age LE"]
    f_scaler = result["Feature Scaler"]

    try:
        joblib.dump(status_le, "./data/preprocessors/" + str(timestamp) + "_statusLE.pkl")
        logger.info("status_le Saved!")

        joblib.dump(age_le, "./data/preprocessors/" + str(timestamp) + "_ageLE.pkl")
        logger.info("age_le Saved!")

        joblib.dump(f_scaler, "./data/preprocessors/" + str(timestamp) + "_fScaler.pkl")
        logger.info("f_scaler Saved!")

        context.log_event(
            AssetMaterialization(
                asset_key="status_le",
                description="Persisted status_le to storage."
            )
        )

        context.log_event(
            AssetMaterialization(
                asset_key="age_le",
                description="Persisted age_le to storage."
            )
        )

        context.log_event(
            AssetMaterialization(
                asset_key="f_scaler",
                description="Persisted f_scaler to storage."
            )
        )
    except Exception as _:
        logger.error("Could not save Preprocessors!")
        raise Exception("Could not save Preprocessors!")

    yield Output(
        value=result["targets"],
        output_name="targets"
    )

    yield Output(
        value=result["features"],
        output_name="features"
    )

    yield Output(
        value=status_le,
        output_name="status_le"
    )

    yield Output(
        value=age_le,
        output_name="age_le"
    )

    yield Output(
        value=f_scaler,
        output_name="f_scaler"
    )


@op(out={
    "targets": Out(),
    "features": Out()
})
def test_split_transformation(df, status_le, age_le, f_scaler):
    result = test_split_transformation_(df, status_le, age_le, f_scaler)

    yield Output(
        value=result["targets"],
        output_name="targets"
    )

    yield Output(
        value=result["features"],
        output_name="features"
    )


@op(out={
    "model": Out(),
    "parameters": Out()
})
def model_training(context, targets, features):
    now = datetime.now()
    timestamp = now.timestamp()
    logger = get_dagster_logger()

    result = model_training_(targets, features)
    model = result["model"]

    try:
        joblib.dump(model, "./data/preprocessors/" + str(timestamp) + ".pkl")
        logger.info("elasticNetModel Saved!")

        context.log_event(
            AssetMaterialization(
                asset_key="elasticNet",
                description="Persisted elasticNet to storage."
            )
        )

    except Exception as _:
        logger.error("Could not save model!")
        raise Exception("Could not save model!")

    yield Output(
        value=result["model"],
        output_name="model"
    )

    yield Output(
        value=result["parameters"],
        output_name="parameters"
    )


@op
def model_evaluation(model, features, targets, params):
    model_evaluation_(model, features, targets, params)


@job
def ml_pipeline():
    setup = setup_folders()

    df = download_data(setup)

    df_clean = data_validation(df)

    train_split, test_split = train_test_split(df_clean)

    result = train_split_transformation(train_split)

    train_targets, train_features, status_le, age_le, f_scaler = result

    result = test_split_transformation(
        test_split,
        status_le,
        age_le,
        f_scaler
    )

    test_targets, test_features = result

    result = model_training(targets=train_targets, features=train_features)

    model, params = result

    model_evaluation(
        model=model,
        targets=test_targets,
        features=test_features,
        params=params
    )