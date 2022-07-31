from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression

from ml_models.config.constants import RANDOM_STATE, N_JOBS
from ml_models.config.logger import get_logger
from ml_models.experiment import execute_experiment
from ml_models.load import load_data


def main() -> None:
    logger = get_logger(__name__)

    logger.debug('Loading data')
    X, y = load_data()
    logger.debug('Data loaded')

    # ==================== ==================== ==================== ==================== ====================
    # ROS + LogisticRegression
    experiment_id = 'ROS + LogReg'
    logger.debug('[{0}] Starting experiment'.format(experiment_id))
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=N_JOBS)
    execute_experiment(experiment_id, X, y, ros, log_reg)
    logger.debug('[{0}] Experiment finished'.format(experiment_id))

    # ==================== ==================== ==================== ==================== ====================
    # RUS + LogisticRegression
    experiment_id = 'RUS + LogReg'
    logger.debug('[{0}] Starting experiment'.format(experiment_id))
    ros = RandomUnderSampler(random_state=RANDOM_STATE)
    log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=N_JOBS)
    execute_experiment(experiment_id, X, y, ros, log_reg)
    logger.debug('[{0}] Experiment finished'.format(experiment_id))

    # ==================== ==================== ==================== ==================== ====================
    # SMOTE + LogisticRegression
    experiment_id = 'SMOTE + LogReg'
    logger.debug('[{0}] Starting experiment'.format(experiment_id))
    smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE, k_neighbors=5, n_jobs=N_JOBS)
    log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=N_JOBS)
    execute_experiment(experiment_id, X, y, smote, log_reg)
    logger.debug('[{0}] Experiment finished'.format(experiment_id))


if __name__ == '__main__':
    main()
