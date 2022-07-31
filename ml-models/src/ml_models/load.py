from typing import Tuple

import pandas as pd
from pandas import Series, DataFrame

from ml_models.config.constants import DATA_PATH


def load_data(data_path: str = DATA_PATH) -> Tuple[DataFrame, Series]:
    df = pd.read_csv(data_path)
    return df.drop(columns=['Class']), df['Class']
