import React from 'react';
import FilterCheck from './FilterCheck';

const Filters = (props) => {
  return (
    <div className='text-center'>
      <div className='btn-group me-2' >
        <FilterCheck setFilters={props.setFilters} id={'ROS'} />
        <FilterCheck setFilters={props.setFilters} id={'RUS'} />
        <FilterCheck setFilters={props.setFilters} id={'SMOTE'} />
      </div>

      <div className='btn-group me-2' >
        <FilterCheck setFilters={props.setFilters} id={'LR'} name={'Logistic Regression'} />
        <FilterCheck setFilters={props.setFilters} id={'RFC'} name={'Random Forest'} />
      </div>
    </div>
  );
};

export default Filters;