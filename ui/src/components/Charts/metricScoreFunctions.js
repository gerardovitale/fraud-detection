import { styles } from './chartStyles';


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


export const getMetricScoresPerSamplingStartegyByMetric = (rawData, metric) => {
  const metricIndex = rawData.columns.indexOf(metric);
  const expIndex = rawData.columns.indexOf('exp_id');
  const expIDs = rawData.data.map(eachRecord => eachRecord[expIndex])
    .filter((eachRecord, index, arr) => arr.indexOf(eachRecord) === index);
  return expIDs.map((expId, index) => {
    return {
      label: expId,
      data: rawData.data.filter(eachRecord => eachRecord[expIndex] == expId)
        .map(eachRecord => eachRecord[metricIndex]),
      borderWidth: 1,
      borderColor: styles.color.solids[index],
      backgroundColor: styles.color.alphas[index],
    };
  });
};


const getMean = (eachRecord) => {
  return meanTestMetrics.map(meanTestMetric => eachRecord[meanTestMetric]
    .reduce((a, b) => a + b, 0) / eachRecord[meanTestMetric].length);
};


export const applyFilters = (filters, experiment) => {
  return filters.length > 0 ?
    filters.map(filter => experiment.includes(filter)).reduce((a, b) => a || b) :
    false;
};

