function getTestMetricScoresRecordsByExpID(rawData, expID) {
    const expIDs = rawData.filter(eachRecord => eachRecord.message.data.exp_id.includes(expID))
        .map(eachRecord => eachRecord.message.data.exp_id)
    return expIDs.map((expId, index) => {
        return {
            label: expId,
            data: rawData
                .filter(eachRecord => eachRecord.message.data.exp_id === expId)
                .map(eachRecord => {
                    return [
                        eachRecord.message.data.mean_test_precision.reduce((a, b) => a + b, 0) / eachRecord.message.data.mean_test_precision.length,
                        eachRecord.message.data.mean_test_recall.reduce((a, b) => a + b, 0) / eachRecord.message.data.mean_test_recall.length,
                        eachRecord.message.data.mean_test_specificity.reduce((a, b) => a + b, 0) / eachRecord.message.data.mean_test_specificity.length,
                        eachRecord.message.data.mean_test_f1.reduce((a, b) => a + b, 0) / eachRecord.message.data.mean_test_f1.length,
                        eachRecord.message.data.mean_test_geometric_mean_score.reduce((a, b) => a + b, 0) / eachRecord.message.data.mean_test_geometric_mean_score.length,
                        eachRecord.message.data.mean_test_roc_auc.reduce((a, b) => a + b, 0) / eachRecord.message.data.mean_test_roc_auc.length,
                    ]
                })[0],
            borderWidth: 1,
            borderColor: styles.color.solids[index],
            backgroundColor: styles.color.alphas[index]
        }
    })
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
    })
}
