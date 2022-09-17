import { styles } from '../chartStyles';


export const meanTestMetrics = [
  'mean_test_precision', 'mean_test_recall', 'mean_test_specificity',
  'mean_test_f1', 'mean_test_geometric_mean_score', 'mean_test_roc_auc',
];

export const getMeanTestMetricScores = (rawData) => {
  const data = rawData.map(eachRecord => eachRecord.message.data);
  return data.map((eachRecord, index) => {
    return {
      label: eachRecord.exp_id,
      data: getMean(eachRecord),
      borderWidth: 1,
      borderColor: styles.color.solids[index],
      backgroundColor: styles.color.alphas[index]
    };
  });
};

const getMean = (eachRecord) => {
  return meanTestMetrics.map(meanTestMetric => eachRecord[meanTestMetric]
    .reduce((a, b) => a + b, 0) / eachRecord[meanTestMetric].length);
};
