from numpy import linspace, logspace

# paths
DATA_PATH = '/app/data/creditcard.csv'
TEST_DATA_PATH = '/app/data/test_data.csv'
LOGGER_CONFIG_FILE = '/app/ml_models/config/log_config.yml'

RANDOM_STATE = 42
TEST_SIZE = 0.33

# ml model params
MAX_DEPTH = 3
ITER = 2500
N_JOBS = -1

# cross validation params
CV_N_SPLITS = 5
CV_N_REPEATS = 2

# grid search params
GSCV_SS_FLOAT_RATIO = linspace(0.1, 1, 10)
GSCV_LR_C = logspace(-3, 1.1, 10)
GSCV_CNN_N_NEIGHBORS = [1, 3, 5]
GSCV_RFC_MAX_DEPTH = [2, 3, 5, 7]
