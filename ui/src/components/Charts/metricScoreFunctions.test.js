import { applyFilters, getMeanTestMetricScores, getMetricScoresPerSamplingStartegyByMetric, meanTestMetrics } from './metricScoreFunctions';
import { testDataForMeanTestMetricScores, testDataForSamplingStrategy } from './testData';

describe('metricScoreFunctions', () => {

  describe('getMeanTestMetricScores', () => {
    it('it should get data successfully without filters', () => {
      const actualOutput = getMeanTestMetricScores(testDataForMeanTestMetricScores);

      expect(actualOutput).toHaveLength(1);
      expect(actualOutput[0].label).toBe('test_exp_id');
      expect(actualOutput[0].data).toStrictEqual([0.475, 0.5, 0, 0.375, 0.45, 0.755]);
      expect(actualOutput[0].data).toHaveLength(meanTestMetrics.length);
    });
  });

  describe('getMetricScoresPerSamplingStartegy', () => {
    it('it should get data successfully', () => {
      const expectedData = {
        testExp0: [0.55, 0.25, 0.25, 0.10, 0.15, 0.50, 0.75, 0.0, 0.1, 0.8],
        testExp1: [0.82, 0.82, 0.80, 0.78, 0.76, 0.75, 0.73, 0.71, 0.68, 0.65],
      };

      const actualOutput = getMetricScoresPerSamplingStartegyByMetric(
        testDataForSamplingStrategy, 'mean_test_f1');

      expect(actualOutput).toHaveLength(2);
      expect(actualOutput[0].label).toBe('test_exp_0');
      expect(actualOutput[0].data).toStrictEqual(expectedData.testExp0);
      expect(actualOutput[0].data).toHaveLength(expectedData.testExp0.length);
      expect(actualOutput[1].label).toBe('test_exp_1');
      expect(actualOutput[1].data).toStrictEqual(expectedData.testExp1);
      expect(actualOutput[1].data).toHaveLength(expectedData.testExp1.length);
    });

  });

  describe('applyFilters', () => {
    it('it should filter when experiment match', () => {
      const filters = ['ROS'];
      const result = applyFilters(filters, 'ROS + RanFor + Grid');
      expect(result).toBe(true);

    });
    it('it should not filter when filters are empty', () => {
      const filters = [];
      const result = applyFilters(filters, 'RUS');
      expect(result).toBe(false);

    });
    it('it should not filter when filters do not match experiments ', () => {
      const filters = ['None'];
      const result = applyFilters(filters, 'Grid');
      expect(result).toBe(false);

    });

  });

});