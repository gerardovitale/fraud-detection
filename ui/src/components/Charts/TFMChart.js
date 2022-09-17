import {
  BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, LineElement, PointElement, Title,
  Tooltip
} from 'chart.js';
import React, { useEffect, useState } from 'react';
import { Bar, Line } from 'react-chartjs-2';


ChartJS.register(
  CategoryScale,
  PointElement,
  LinearScale,
  BarElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);


const TFMChart = (props) => {

  const [chartData, setChartData] = useState({});
  const [chartOptions, setChartOptions] = useState({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      const request = new Request(props.url, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
        cache: 'default',
      });

      await fetch(request)
        .then(response => response.json())
        .then(rawData => {
          const labels = props.labels;
          const data = {
            labels,
            datasets: 'metric' in props ?
              props.dataProcessor(rawData.result, props.metric) :
              props.dataProcessor(rawData.result),
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

  if (props.chartType === 'bar') {
    return (
      <div>
        <h4>{props.title}</h4>
        <Bar data={chartData} options={chartOptions} />
      </div>
    );
  }

  if (props.chartType === 'line') {
    return (
      <div>
        <h4>{props.title}</h4>
        <Line data={chartData} options={chartOptions} />
      </div>
    );
  }

};

export default TFMChart;
