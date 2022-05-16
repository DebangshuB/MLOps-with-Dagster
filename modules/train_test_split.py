from dask_ml.model_selection import train_test_split


def train_test_split_(df):
    train, test = train_test_split(
        df, test_size=0.2, random_state=42
    )

    return {
        "train_split": train,
        "test_split": test
    }