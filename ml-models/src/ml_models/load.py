import pandas as pd

from ml_models.config.constants import DATA_PATH


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df.drop(columns=['Class']), df['Class']
