from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import (CondensedNearestNeighbour,
                                     RandomUnderSampler)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from ml_models.config.constants import (GSCV_CNN_N_NEIGHBORS, GSCV_LR_C,
                                        GSCV_RFC_MAX_DEPTH,
                                        GSCV_SS_FLOAT_RATIO, ITER, N_JOBS,
                                        CV_N_REPEATS, CV_N_SPLITS, RANDOM_STATE)
from ml_models.config.logger import get_logger
from ml_models.experiment import exec_grid_exp
from ml_models.load import load_data


def main() -> None:
    logger = get_logger(__name__)

    logger.debug('Loading data')
    X, y = load_data()
    logger.debug('Data loaded')

    # resampling strategies
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=5, n_jobs=N_JOBS)
    cnn = CondensedNearestNeighbour(random_state=RANDOM_STATE, n_jobs=N_JOBS)

    # estimators
    lrc = LogisticRegression(random_state=RANDOM_STATE, max_iter=ITER, n_jobs=N_JOBS)
    rfc = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)

    # cross validation strategy
    cvs = RepeatedStratifiedKFold(
        n_splits=CV_N_SPLITS, n_repeats=CV_N_REPEATS, random_state=RANDOM_STATE)

    # grid params
    grid_param_lrc = {
        'C': GSCV_LR_C,
    }
    grid_param_res_lrc = {
        'resampling__sampling_strategy': GSCV_SS_FLOAT_RATIO,
        'estimator__C': GSCV_LR_C,
    }
    grid_param_cnn_lrc = {
        'resampling__n_neighbors': GSCV_CNN_N_NEIGHBORS,
        'estimator__C': GSCV_LR_C,
    }
    grid_param_rfc = {
        'max_depth': GSCV_RFC_MAX_DEPTH,
    }
    grid_param_res_rfc = {
        'resampling__sampling_strategy': GSCV_SS_FLOAT_RATIO,
        'estimator__max_depth': GSCV_RFC_MAX_DEPTH,
    }
    grid_param_cnn_rfc = {
        'resampling__n_neighbors': GSCV_CNN_N_NEIGHBORS,
        'estimator__max_depth': GSCV_RFC_MAX_DEPTH,
    }

    # experiment_id standard model => <resampling_strategy> + <ml_model> + <additional_method>

    # experiments using a grid search cv strategy
    exec_grid_exp('None + LR + GSCV', X, y, None, lrc, cvs, grid_param_lrc)
    exec_grid_exp('ROS + LR + GSCV', X, y, ros, lrc, cvs, grid_param_res_lrc)
    exec_grid_exp('RUS + LR + GSCV', X, y, rus, lrc, cvs, grid_param_res_lrc)
    exec_grid_exp('SMOTE + LR + GSCV', X, y, smote, lrc, cvs, grid_param_res_lrc)
    exec_grid_exp('CNN + LR + GSCV', X, y, cnn, lrc, cvs, grid_param_cnn_lrc)
    
    exec_grid_exp('None + RFC + GSCV', X, y, None, rfc, cvs, grid_param_rfc)
    exec_grid_exp('ROS + RFC + GSCV', X, y, ros, rfc, cvs, grid_param_res_rfc)
    exec_grid_exp('RUS + RFC + GSCV', X, y, rus, rfc, cvs, grid_param_res_rfc)
    exec_grid_exp('SMOTE + RFC + GSCV', X, y, smote, rfc, cvs, grid_param_res_rfc)
    exec_grid_exp('CNN + RFC + GSCV', X, y, cnn, rfc, cvs, grid_param_cnn_rfc)


if __name__ == '__main__':
    main()
