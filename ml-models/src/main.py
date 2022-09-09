from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from numpy import linspace, logspace
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from ml_models.config.constants import (ITER, N_JOBS, N_REPEATS, N_SPLITS,
                                        RANDOM_STATE)
from ml_models.config.logger import get_logger
from ml_models.experiment import exec_cross_exp, exec_exp, exec_grid_exp
from ml_models.load import load_data


def main() -> None:
    logger = get_logger(__name__)

    logger.debug('Loading data')
    X, y = load_data()
    logger.debug('Data loaded')

    # resampling strategies
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    rus = RandomUnderSampler(random_state=RANDOM_STATE)
    smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE, k_neighbors=5, n_jobs=N_JOBS)

    # estimators
    lrc = LogisticRegression(random_state=RANDOM_STATE, max_iter=ITER, n_jobs=N_JOBS)
    rfc = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)

    # cross validation strategy
    cvs = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    # grid params
    grid_param_lrc = {
        'C': logspace(-3, 1.1, 10),
    }
    grid_param_res_lrc = {
        'resampling__sampling_strategy': linspace(0.1, 1, 10),
        'estimator__C': logspace(-3, 1.1, 10),
    }
    grid_param_rfc = {
        'max_depth': [2, 3, 5, 7],
    }
    grid_param_res_rfc = {
        'resampling__sampling_strategy': linspace(0.1, 1, 10),
        'estimator__max_depth': [2, 3, 5, 7],
    }

    # experiment_id standard model => <resampling_strategy> + <ml_model> + <additional_method>

    # ==================== ==================== ==================== ==================== ====================
    # basic experiments
    exec_exp('ROS + LogReg', X, y, ros, lrc)
    exec_exp('RUS + LogReg', X, y, rus, lrc)
    exec_exp('SMOTE + LogReg', X, y, smote, lrc)
    exec_exp('ROS + RanFor', X, y, ros, rfc)
    exec_exp('RUS + RanFor', X, y, rus, rfc)
    exec_exp('SMOTE + RanFor', X, y, smote, rfc)

    # ==================== ==================== ==================== ==================== ====================
    # experiments using a cross validation strategy
    exec_cross_exp('None + LogReg + Cross', X, y, None, lrc, cvs)
    exec_cross_exp('ROS + LogReg + Cross', X, y, ros, lrc, cvs)
    exec_cross_exp('RUS + LogReg + Cross', X, y, rus, lrc, cvs)
    exec_cross_exp('SMOTE + LogReg + Cross', X, y, smote, lrc, cvs)
    exec_cross_exp('None + RanFor + Cross', X, y, None, rfc, cvs)
    exec_cross_exp('ROS + RanFor + Cross', X, y, ros, rfc, cvs)
    exec_cross_exp('RUS + RanFor + Cross', X, y, rus, rfc, cvs)
    exec_cross_exp('SMOTE + RanFor + Cross', X, y, smote, rfc, cvs)

    # ==================== ==================== ==================== ==================== ====================
    # experiments using a grid search cv strategy
    exec_grid_exp('None + LogReg + Grid', X, y, None, lrc, cvs, grid_param_lrc)
    exec_grid_exp('ROS + LogReg + Grid', X, y, ros, lrc, cvs, grid_param_res_lrc)
    exec_grid_exp('RUS + LogReg + Grid', X, y, rus, lrc, cvs, grid_param_res_lrc)
    exec_grid_exp('SMOTE + LogReg + Grid', X, y, smote, lrc, cvs, grid_param_res_lrc)
    exec_grid_exp('None + RanFor + Grid', X, y, None, rfc, cvs, grid_param_rfc)
    exec_grid_exp('ROS + RanFor + Grid', X, y, ros, rfc, cvs, grid_param_res_rfc)
    exec_grid_exp('RUS + RanFor + Grid', X, y, rus, rfc, cvs, grid_param_res_rfc)
    exec_grid_exp('SMOTE + RanFor + Grid', X, y, smote, rfc, cvs, grid_param_res_rfc)


if __name__ == '__main__':
    main()
