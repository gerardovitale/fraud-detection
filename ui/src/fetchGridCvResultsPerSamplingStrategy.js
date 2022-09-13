const gridCvResultsPerSamplingStrategyRequest = new Request('http://localhost:8080/grid_data_per_sampling_strategy.json', {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
    mode: 'cors',
    cache: 'default',
});

fetch(gridCvResultsPerSamplingStrategyRequest)
    .then(response => response.json())
    .then(data => displayMetricsPerSamplingStrategyCharts(data))

    .catch(err => {
        throw err;
    })
