from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from ml_models.config.constants import RANDOM_STATE, N_JOBS
from ml_models.config.logger import get_logger
from ml_models.experiment import execute_experiment
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
    log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=N_JOBS)
    rfc = RandomForestClassifier(max_depth=2, random_state=RANDOM_STATE)
    
    ## experiment_id standard model => <resampling_strategy> + <ml_model> + <additional_method>

    # ==================== ==================== ==================== ==================== ====================
    # ROS + LogisticRegression
    experiment_id = 'ROS + LogReg'
    logger.debug('[{0}] Starting experiment'.format(experiment_id))
    execute_experiment(experiment_id, X, y, ros, log_reg)
    logger.debug('[{0}] Experiment finished'.format(experiment_id))

    # ==================== ==================== ==================== ==================== ====================
    # RUS + LogisticRegression
    experiment_id = 'RUS + LogReg'
    logger.debug('[{0}] Starting experiment'.format(experiment_id))
    execute_experiment(experiment_id, X, y, rus, log_reg)
    logger.debug('[{0}] Experiment finished'.format(experiment_id))

    # ==================== ==================== ==================== ==================== ====================
    # SMOTE + LogisticRegression
    experiment_id = 'SMOTE + LogReg'
    logger.debug('[{0}] Starting experiment'.format(experiment_id))
    execute_experiment(experiment_id, X, y, smote, log_reg)
    logger.debug('[{0}] Experiment finished'.format(experiment_id))

    # ==================== ==================== ==================== ==================== ====================
    # ROS + RandomForest
    experiment_id = 'ROS + RanFor'
    logger.debug('[{0}] Starting experiment'.format(experiment_id))
    execute_experiment(experiment_id, X, y, ros, rfc)
    logger.debug('[{0}] Experiment finished'.format(experiment_id))

    # ==================== ==================== ==================== ==================== ====================
    # RUS + RandomForest
    experiment_id = 'RUS + RanFor'
    logger.debug('[{0}] Starting experiment'.format(experiment_id))
    execute_experiment(experiment_id, X, y, rus, rfc)
    logger.debug('[{0}] Experiment finished'.format(experiment_id))

    # ==================== ==================== ==================== ==================== ====================
    # SMOTE + RandomForest
    experiment_id = 'SMOTE + RanFor'
    logger.debug('[{0}] Starting experiment'.format(experiment_id))
    execute_experiment(experiment_id, X, y, smote, rfc)
    logger.debug('[{0}] Experiment finished'.format(experiment_id))


if __name__ == '__main__':
    main()
