from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


def test_split_transformation_(df, status_le, age_le, feature_scaler):
    # Price
    # Converting everything to lacs
    df["Price"] = pd.to_numeric((df["Price"]))
    df.loc[df["Unit"] == "Cr", "Price"] = df["Price"] * 100
    df.drop("Unit", axis=1, inplace=True)

    # Status
    df["Status"] = status_le.transform(df["Status"])

    # Age
    df["Age"] = age_le.transform(df["Age"])

    targets = df["Price"].to_numpy()
    features = df.drop("Price", axis=1).to_numpy()

    # Scaling the features
    scaled_features = feature_scaler.transform(features)

    return {
        "targets": targets,
        "features": scaled_features
    }