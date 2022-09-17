import React from 'react';
import './App.css';
import TFMChart from './components/Charts/TFMChart';


function App() {
  return (
    <figure>
      <h3>Mean Test Metric Scores</h3>
      <TFMChart />
    </figure>
  );
}

export default App;
