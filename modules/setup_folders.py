import os


def setup_folders_():
    folders = [
        "./data",
        "./data/models",
        "./data/csv",
        "./data/preprocessors"
    ]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)