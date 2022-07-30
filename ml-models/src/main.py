from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.linear_model import LogisticRegression

from ml_models.config.constants import RANDOM_STATE
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
    logger.debug('Starting experiment')
    ros = RandomOverSampler(random_state=RANDOM_STATE)
    log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1)
    execute_experiment('ROS + LogReg', X, y, ros, log_reg)
    logger.debug('Experiment finished')

    # ==================== ==================== ==================== ==================== ====================
    # SMOTE + LogisticRegression
    # logger.debug('Starting experiment')
    # smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE, k_neighbors=5, n_jobs=-1)
    # log_reg = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, n_jobs=-1)
    # execute_experiment('SMOTE + LogReg', X, y, smote, log_reg)
    # logger.debug('Experiment finished')


if __name__ == '__main__':
    main()
