from datetime import datetime
import pandas as pd


def model_evaluation_(model, features, targets, params):
    now = datetime.now()
    timestamp = now.timestamp()

    # Load model performance history
    try:
        model_history = pd.read_csv("./data/model_history.csv")
    except (FileNotFoundError, pd.errors.EmptyDataError) as _:
        with open("./data/model_history.csv", "w") as file_out:
            file_out.write("timestamp,model_name,score,alpha,l1_ratio,selection")

        model_history = pd.read_csv("./data/model_history.csv")

    # Scoring the current model
    targets = targets.reshape(-1, 1)
    score_ = model.score(features, targets)

    addition = {
        "timestamp": [timestamp],
        "model_name": [str(timestamp) + ".pkl"],
        "score": [score_],
        "alpha": [params["alpha"]],
        "l1_ratio": [params["l1_ratio"]],
        "selection": [params["selection"]]
    }

    addition = pd.DataFrame.from_dict(addition)

    model_history = pd.concat([model_history, addition], ignore_index=True, axis=0)

    model_history.to_csv("./data/model_history.csv")