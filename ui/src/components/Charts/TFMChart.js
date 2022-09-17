import {
  BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Title,
  Tooltip
} from 'chart.js';
import React, { useEffect, useState } from 'react';
import { Bar } from 'react-chartjs-2';
import { getMeanTestMetricScores } from './metricScoreFunctions';


ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);


const TFMChart = () => {

  const [chartData, setChartData] = useState({});
  const [chartOptions, setChartOptions] = useState({});
  const [loading, setLoading] = useState(true);

  const url = 'http://localhost:8080/grid_cv_results_model_data.json';
  const labels = ['Precision', 'Recall', 'Specificity', 'F-Score', 'G-Score', 'ROC AUC'];

  useEffect(() => {
    const fetchData = async () => {
      const request = new Request(url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
        cache: 'default',
      });

      await fetch(request)
        .then(response => response.json())
        .then(rawData => {
          console.log(rawData);
          const data = {
            labels,
            datasets: getMeanTestMetricScores(rawData.result),
          };
          setChartData(data);
          setLoading(false);
        })
        .catch((error) => {
          console.log('error', error);
        });
    };

    fetchData();
    setChartOptions({
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
      }
    });
  }, []);

  if (loading) {
    return (<p>Loading...</p>);
  }

  return (
    <Bar data={chartData} options={chartOptions} />
  );
};

export default TFMChart;
