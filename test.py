from modules.get_data import fetch_data_
from modules.data_validation import validate_data_
from modules.train_test_split import train_test_split_
from modules.train_split_transformation import train_split_transformation_
from modules.model_evaluation import model_evaluation_
from modules.model_training import model_training_
from modules.test_split_transformation import test_split_transformation_
from modules.setup_folders import setup_folders_

setup_folders_()

df = fetch_data_()

df = validate_data_(df)["df_clean"]

result = train_test_split_(df)

train_split = result["train_split"]
test_split = result["test_split"]

result = train_split_transformation_(train_split)

train_targets = result["targets"]
train_features = result["features"]
status_le = result["Status LE"]
age_le = result["Age LE"]
f_scaler = result["Feature Scaler"]

result = test_split_transformation_(
    test_split,
    status_le,
    age_le,
    f_scaler
)

test_targets = result["targets"]
test_features = result["features"]

result = model_training_(targets=train_targets, features=train_features)

model = result["model"]
params = result["parameters"]

model_evaluation_(
    model=model,
    targets=test_targets,
    features=test_features,
    params=params
)