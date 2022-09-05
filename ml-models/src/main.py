from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from numpy import logspace
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from ml_models.config.constants import N_JOBS, N_REPEATS, N_SPLITS, RANDOM_STATE
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
    smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE, k_neighbors=5, n_jobs=N_JOBS)

    # estimators
    lrc = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=N_JOBS)
    rfc = RandomForestClassifier(max_depth=2, random_state=RANDOM_STATE)

    # cross validation strategy
    cvs = RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)

    # grid params
    grid_param_lrc = {'C': logspace(-3, 1.1, 10),}
    grid_param_res_lrc = {
        'resampling__sampling_strategy': ['minority', 'not minority', 'not majority', 'all'],
        'estimator__C': logspace(-3, 1.1, 10)
    }

    # experiment_id standard model => <resampling_strategy> + <ml_model> + <additional_method>
    exec_grid_exp('None + LogReg + Grid', X, y, None, lrc, cvs, grid_param_lrc)
    exec_grid_exp('ROS + LogReg + Grid', X, y, ros, lrc, cvs, grid_param_res_lrc)
    exec_grid_exp('RUS + LogReg + Grid', X, y, rus, lrc, cvs, grid_param_res_lrc)
    exec_grid_exp('SMOTE + LogReg + Grid', X, y, smote, lrc, cvs, grid_param_res_lrc)


if __name__ == '__main__':
    main()
