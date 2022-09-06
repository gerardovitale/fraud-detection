from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from imblearn.metrics import geometric_mean_score

# paths
DATA_PATH = '/app/data/creditcard.csv'
TEST_DATA_PATH = '/app/data/test_data.csv'
LOGGER_CONFIG_FILE = '/app/ml_models/config/log_config.yml'

RANDOM_STATE = 42
TEST_SIZE = 0.33

# ml model params
MAX_DEPTH = 3
ITER = 3500
N_JOBS = -1

# cross validation params
N_SPLITS = 5
N_REPEATS = 2

# grid params
SCORING = {
    'f1': make_scorer(f1_score, greater_is_better=True),
    'recall': make_scorer(recall_score, greater_is_better=True),
    'precision': make_scorer(precision_score, greater_is_better=True),
    'roc_auc': make_scorer(roc_auc_score, greater_is_better=True),
    'geometric_mean_score': make_scorer(geometric_mean_score, greater_is_better=True),
}
