import 'bootstrap/dist/css/bootstrap.min.css';
import React from 'react';
import './App.css';
import { getMeanTestMetricScores, getMetricScoresPerSamplingStartegyByMetric } from './components/Charts/metricScoreFunctions';
import TFMChart from './components/Charts/TFMChart';

function App() {
  const meanTestScoreUrl = 'http://localhost:8080/grid_cv_results_model_data.json';
  const meanTestScoreLabels = ['Precision', 'Recall', 'Specificity', 'F-Score', 'G-Score', 'ROC AUC'];

  const samplingStrategyUrl = 'http://localhost:8080/grid_data_per_sampling_strategy.json';
  const samplingStrategyLabels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
  const meanTestMetrics = [
    'mean_test_precision', 'mean_test_recall', 'mean_test_specificity',
    'mean_test_f1', 'mean_test_geometric_mean_score', 'mean_test_roc_auc',
  ];
  return (
    <div>
      <h2>Overall Metric Scores</h2>
      <section>
        <figure>
          <TFMChart
            title={'Mean Test Scores per Experiment'}
            url={meanTestScoreUrl}
            labels={meanTestScoreLabels}
            chartType={'bar'}
            dataProcessor={getMeanTestMetricScores}
          />
        </figure>
      </section>

      <h2>Mean Test Scores Variation Vs. Sampling Strategy Float Ratio Variation</h2>
      <section className='row'>
        {meanTestMetrics.map((metric, index) => (
          // eslint-disable-next-line react/jsx-key
          <figure className='col-4'>
            <TFMChart
              title={meanTestScoreLabels[index]}
              url={samplingStrategyUrl}
              labels={samplingStrategyLabels}
              chartType={'line'}
              dataProcessor={getMetricScoresPerSamplingStartegyByMetric}
              metric={metric}
            />
          </figure>
        ))}
      </section>
    </div>
  );
}

export default App;
