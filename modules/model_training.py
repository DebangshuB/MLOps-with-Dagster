from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV


def model_training_(targets, features):
    model = ElasticNet(
        random_state=42
    )

    param_space = {
        "l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
        "alpha": [0.2, 0.7, 1],
        "selection": ["cyclic", "random"]
    }

    search = GridSearchCV(model, param_space)

    search.fit(features, targets)

    return {
        "model": search.best_estimator_,
        "parameters": search.best_params_
    }