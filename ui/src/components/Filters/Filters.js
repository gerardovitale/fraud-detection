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
        <FilterCheck setFilters={props.setFilters} id={'LogReg'} name={'Logistic Regression'} />
        <FilterCheck setFilters={props.setFilters} id={'RanFor'} name={'Random Forest'} />
      </div>
    </div>
  );
};

export default Filters;