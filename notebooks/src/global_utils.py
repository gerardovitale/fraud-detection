import os
from typing import Any, Dict

from seaborn import palettes


def get_globals() -> Dict[str, Any]:
    if 'google.colab' in str(get_ipython()):
        from google.colab import drive
        drive.mount('/content/drive')
        data_path = 'drive/MyDrive/BigDataMaster/TFM/data/creditcard.csv'
    elif os.getenv('CONTAINER_BASE_DIR'):
        data_path = '../data/creditcard.csv'
    else:
        data_path = '../../data/creditcard.csv'

    return {
        'DATA_PATH': data_path,
        'COLORS': palettes.color_palette('colorblind'),
        'RANDOM_STATE': 42,
        'TEST_SIZE': 0.33,
        'ITER': 2500,
        'N_JOBS': -1,
    }
