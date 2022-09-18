import {
  BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, LineElement, PointElement, Title,
  Tooltip
} from 'chart.js';
import React, { useEffect, useState } from 'react';
import { Bar, Line } from 'react-chartjs-2';
import { isHidden } from './metricScoreFunctions';

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
    fetch(props.url)
      .then(response => response.json())
      .then(rawData => {
        const labels = props.labels;
        const data = {
          labels,
          datasets: 'metric' in props ?
            props.dataProcessor(rawData.result, props.metric, props.filters) :
            props.dataProcessor(rawData.result, props.filters),
        };
        setChartData(data);
        setLoading(false);
      })
      .catch((error) => {
        console.log('error', error);
      });

    setChartOptions({
      responsive: true,
      plugins: {
        legend: {
          position: 'top',
        },
      }
    });
  }, []);

  useEffect(() => {
    if (!loading) {
      setChartData(prevChartData => ({
        ...prevChartData,
        datasets: prevChartData.datasets.map(
          ds => ({ ...ds, hidden: isHidden(props.filters, ds.label) }))
      }
      ));
    }
  }, [props.filters]);

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
