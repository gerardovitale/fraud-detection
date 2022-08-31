import os
from typing import Any, Dict

from seaborn import palettes

from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from imblearn.metrics import geometric_mean_score


def get_globals() -> Dict[str, Any]:
    # when google colab is used
    if 'google.colab' in str(get_ipython()):
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_PATH = 'drive/MyDrive/BigDataMaster/TFM/data/creditcard.csv'

    # when docker is used
    elif os.getenv('CONTAINER_BASE_DIR'):
        DATA_PATH = '../data/creditcard.csv'

    # when conda is used
    else:
        DATA_PATH = '../../data/creditcard.csv'

    SCORING = {
        'f1': make_scorer(f1_score, greater_is_better=True),
        'recall': make_scorer(recall_score, greater_is_better=True),
        'precision': make_scorer(precision_score, greater_is_better=True),
        'roc_auc': make_scorer(roc_auc_score, greater_is_better=True),
        'geometric_mean_score': make_scorer(geometric_mean_score, greater_is_better=True),
    }

    return {
        'DATA_PATH': DATA_PATH,
        'COLORS': palettes.color_palette('colorblind'),
        'RANDOM_STATE': 42,
        'TEST_SIZE': 0.33,
        'ITER': 2500,
        'N_JOBS': -1,
        'N_SPLITS': 10,
        'N_REPEATS': 3,
        'SCORING': SCORING,
    }
