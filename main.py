from datetime import datetime
from dagster import (
    job, op,
    AssetMaterialization,
    get_dagster_logger
)

from modules.get_data import fetch_data_
from modules.data_validation import validate_data_
from modules.train_test_split import train_test_split_
from modules.train_split_transformation import train_split_transformation_


@op
def download_data():
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


@op
def train_test_split(df):
    result = train_test_split_(df)

    return result


@op
def train_split_transformation(df):
    result = train_split_transformation_(df)

    return result