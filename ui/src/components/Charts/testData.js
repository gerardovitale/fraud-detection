export const testDataForMeanTestMetricScores = [
  {
    message: {
      data: {
        exp_id: 'test_exp_id',
        mean_test_f1: [0.25, 0.50],
        mean_test_geometric_mean_score: [0.15, 0.75],
        mean_test_precision: [0.35, 0.60],
        mean_test_recall: [0.75, 0.25],
        mean_test_roc_auc: [0.75, 0.76],
        mean_test_specificity: [0.0, 0.0]
      },
    },
  },
];

export const testDataForSamplingStrategy = {
  columns: [
    'exp_id', 'param_resampling__sampling_strategy', 'mean_test_f1',
    'mean_test_recall', 'mean_test_precision', 'mean_test_specificity',
    'mean_test_roc_auc', 'mean_test_geometric_mean_score'
  ],
  data: [
    ['test_exp_0', 0.1, 0.55],
    ['test_exp_0', 0.2, 0.25],
    ['test_exp_0', 0.30000000000000004, 0.25],
    ['test_exp_0', 0.4, 0.10],
    ['test_exp_0', 0.5, 0.15],
    ['test_exp_0', 0.6, 0.50],
    ['test_exp_0', 0.7000000000000001, 0.75],
    ['test_exp_0', 0.8, 0.0],
    ['test_exp_0', 0.9, 0.1],
    ['test_exp_0', 1, 0.80],
    ['test_exp_1', 0.1, 0.82],
    ['test_exp_1', 0.2, 0.82],
    ['test_exp_1', 0.30000000000000004, 0.80],
    ['test_exp_1', 0.4, 0.78],
    ['test_exp_1', 0.5, 0.76],
    ['test_exp_1', 0.6, 0.75],
    ['test_exp_1', 0.7000000000000001, 0.73],
    ['test_exp_1', 0.8, 0.71],
    ['test_exp_1', 0.9, 0.68],
    ['test_exp_1', 1, 0.65],
  ],
};
