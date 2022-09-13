Chart.defaults.global.defaultFontColor = '#fff'
Chart.defaults.global.elements.line.borderWidth = 1
Chart.defaults.global.elements.rectangle.borderWidth = 1
Chart.defaults.scale.gridLines.color = '#444'


function displayMeanTestMetricCharts(data) {
    document.body.classList.add('running');

    const metricLabels = ['Precision', 'Recall', 'Specificity', 'F-Score', 'G-Score', 'ROC AUC'];

    const logRegTestMetricScores = getTestMetricScoresRecordsByExpID(data.result, 'LogReg');
    barChart(logRegTestMetricScores, metricLabels, 'LogRegTestMetricScores', 'bar');

    const ranForTestMetricScores = getTestMetricScoresRecordsByExpID(data.result, 'RanFor');
    barChart(ranForTestMetricScores, metricLabels, 'RanForTestMetricScores', 'bar');
}


function displayMetricsPerSamplingStrategyCharts(data) {
    const samplingStartegylabels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];

    const precisionPerSamplingStartegy = getMetricPerSamplingStartegy(data.result, 'mean_test_precision');
    barChart(precisionPerSamplingStartegy, samplingStartegylabels, 'precisionPerSamplingStartegy', 'line');

    const recallPerSamplingStartegy = getMetricPerSamplingStartegy(data.result, 'mean_test_recall');
    barChart(recallPerSamplingStartegy, samplingStartegylabels, 'recallPerSamplingStartegy', 'line');

    const specificityPerSamplingStartegy = getMetricPerSamplingStartegy(data.result, 'mean_test_specificity');
    barChart(specificityPerSamplingStartegy, samplingStartegylabels, 'specificityPerSamplingStartegy', 'line');

    const f1PerSamplingStartegy = getMetricPerSamplingStartegy(data.result, 'mean_test_f1');
    barChart(f1PerSamplingStartegy, samplingStartegylabels, 'f1PerSamplingStartegy', 'line');

    const gMeanPerSamplingStartegy = getMetricPerSamplingStartegy(data.result, 'mean_test_geometric_mean_score');
    barChart(gMeanPerSamplingStartegy, samplingStartegylabels, 'gMeanPerSamplingStartegy', 'line');

    const rocAucPerSamplingStartegy = getMetricPerSamplingStartegy(data.result, 'mean_test_roc_auc');
    barChart(rocAucPerSamplingStartegy, samplingStartegylabels, 'rocAucPerSamplingStartegy', 'line');
}


function barChart(records, labels, id, chartType) {
    const data = {
        labels: labels,
        datasets: records,
    }
    const options = {
        responsive: true,
        legend: {
            position: 'top',
            labels: {
                fontColor: 'grey',
            }
        }
    }
    new Chart(id, { type: chartType, data, options })
}
