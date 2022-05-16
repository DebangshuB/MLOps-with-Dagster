from dask_ml.preprocessing import LabelEncoder, RobustScaler
import dask.array as da


def train_split_transformation_(df):
    # Price
    # Converting everything to lacs
    df.loc[df["Unit"] == "Cr", "Price"] = df["Price"] * 100
    df.drop("Unit", axis=1, inplace=True)

    # Status
    status_le = LabelEncoder()
    df["Status"] = status_le.fit_transform(df["Status"])

    # Age
    age_le = LabelEncoder()
    df["Age"] = age_le.fit_transform(df["Age"])

    targets = da.from_array(df["Price"].to_numpy())
    features = da.from_array(df.drop("Price", axis=1).to_numpy())

    # Scaling the features
    feature_scaler = RobustScaler()
    scaled_features = feature_scaler.fit_transform(features)

    return {
        "targets": targets,
        "features": scaled_features,
        "Status LE": status_le,
        "Age LE": age_le,
        "Feature Scaler": feature_scaler
    }