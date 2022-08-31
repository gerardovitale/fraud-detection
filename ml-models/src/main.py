from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold

from ml_models.config.constants import N_JOBS, RANDOM_STATE, N_SPLITS, N_REPEATS
from ml_models.config.logger import get_logger
from ml_models.experiment import exec_exp, exec_cross_exp
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
    cvs = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=RANDOM_STATE)
    
    ## experiment_id standard model => <resampling_strategy> + <ml_model> + <additional_method>

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
    exec_cross_exp('LogReg + Cross', X, y, None, lrc, cvs)
    exec_cross_exp('ROS + LogReg + Cross', X, y, ros, lrc, cvs)
    exec_cross_exp('RUS + LogReg + Cross', X, y, rus, lrc, cvs)
    exec_cross_exp('SMOTE + LogReg + Cross', X, y, smote, lrc, cvs)

    exec_cross_exp('RanFor + Cross', X, y, None, rfc, cvs)
    exec_cross_exp('ROS + RanFor + Cross', X, y, ros, rfc, cvs)
    exec_cross_exp('RUS + RanFor + Cross', X, y, rus, rfc, cvs)
    exec_cross_exp('SMOTE + RanFor + Cross', X, y, smote, rfc, cvs)


if __name__ == '__main__':
    main()
