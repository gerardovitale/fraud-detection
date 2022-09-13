const gridCvResultsRequest = new Request('http://localhost:8080/grid_cv_results_model_data.json', {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    mode: 'cors',
    cache: 'default',
});

fetch(gridCvResultsRequest)
    .then(response => response.json())
    .then(data => displayMeanTestMetricCharts(data))

    .catch(err => {
        throw err;
    })
