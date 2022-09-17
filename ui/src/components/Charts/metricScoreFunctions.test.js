import { getMeanTestMetricScores, meanTestMetrics } from './metricScoreFunctions';
import { rawData } from './testData';

describe('metricScoreFunctions', () => {

  describe('getMeanTestMetricScores', () => {

    it('it should get data successfully', () => {
      const actualOutput = getMeanTestMetricScores(rawData);

      expect(actualOutput).toHaveLength(1);
      expect(actualOutput[0].label).toBe('None + LogReg + Grid');
      expect(actualOutput[0].data).toStrictEqual([0.475, 0.5, 0, 0.375, 0.45, 0.755]);
      expect(actualOutput[0].data).toHaveLength(meanTestMetrics.length);
    });
  });
});