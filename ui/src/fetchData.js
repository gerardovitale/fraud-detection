function run() {
    const dashboards = {
        urls: [
            'http://localhost:8080/grid_cv_results_model_data.json',
            'http://localhost:8080/grid_data_per_sampling_strategy.json',
        ],
        chartFunctions: [
            displayMeanTestMetricCharts,
            displayMetricsPerSamplingStrategyCharts,
        ]
    }

    dashboards.urls.map((url, index) => {
        const request = new Request(url, {
            method: 'GET',
            headers: { 'Content-Type': 'application/json' },
            mode: 'cors',
            cache: 'default',
        });

        fetch(request)
            .then(response => response.json())
            .then(data => dashboards.chartFunctions[index](data))

            .catch(err => {
                throw err;
            });
    });
}


run()