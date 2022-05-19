from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd


def train_split_transformation_(df):
    # Price
    # Converting everything to lacs
    df["Price"] = pd.to_numeric((df["Price"]))
    df.loc[df["Unit"] == "Cr", "Price"] = df["Price"] * 100
    df.drop("Unit", axis=1, inplace=True)

    # Status
    status_le = LabelEncoder()
    df["Status"] = status_le.fit_transform(df["Status"])

    # Age
    age_le = LabelEncoder()
    df["Age"] = age_le.fit_transform(df["Age"])

    targets = df["Price"].to_numpy()
    features = df.drop("Price", axis=1).to_numpy()

    # Scaling the features
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(features)

    return {
        "targets": targets,
        "features": scaled_features,
        "Status LE": status_le,
        "Age LE": age_le,
        "Feature Scaler": feature_scaler
    }