function getTestMetricScoresRecordsByExpID(rawData, expID) {
    const meanTestMetrics = [
        'mean_test_precision', 'mean_test_recall', 'mean_test_specificity',
        'mean_test_f1', 'mean_test_geometric_mean_score', 'mean_test_roc_auc',
    ];
    const data = rawData.map(eachRecord => eachRecord.message.data);
    const expIDs = data.filter(eachRecord => eachRecord.exp_id.includes(expID))
        .map(eachRecord => eachRecord.exp_id);

    return expIDs.map((expId, index) => {
        return {
            label: expId,
            data: data.filter(eachRecord => eachRecord.exp_id === expId)
                .map(eachRecord => {
                    return meanTestMetrics.map(meanTestMetric =>
                        eachRecord[meanTestMetric].reduce((a, b) =>
                            a + b, 0) / eachRecord.mean_test_precision.length)
                })[0],
            borderWidth: 1,
            borderColor: styles.color.solids[index],
            backgroundColor: styles.color.alphas[index]
        }
    });
}


function getMetricPerSamplingStartegy(rawData, metricName) {
    const f1Index = rawData.columns.indexOf(metricName)
    const expIndex = rawData.columns.indexOf('exp_id')
    const expIDs = rawData.data.map(eachRecord => eachRecord[expIndex])
        .filter((eachRecord, index, arr) => arr.indexOf(eachRecord) === index)
    return expIDs.map((expId, index) => {
        return {
            label: expId,
            data: rawData.data.filter(eachRecord => eachRecord[expIndex] == expId)
                .map(eachRecord => eachRecord[f1Index]),
            borderWidth: 1,
            borderColor: styles.color.solids[index],
            backgroundColor: styles.color.alphas[index],
        }
    });
}
