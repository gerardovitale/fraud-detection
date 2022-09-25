import React from 'react';

const FilterCheck = (props) => {

  const updateFilter = (elem) => {
    if (elem.checked) {
      removeFilter(elem.id);
    } else {
      props.setFilters(filters => [...filters, elem.id]);

    }
  };

  const removeFilter = (elem) => {
    props.setFilters(filters => filters.filter((filter) => filter != elem));
  };

  return (
    <>
      <input type='checkbox' className='btn-check' id={props.id} defaultChecked={true} 
        onClick={(e) => {
          updateFilter(e.target);
        }}
      />
      <label className='btn btn-outline-dark' htmlFor={props.id} >{props.name ? props.name : props.id}</label>
    </>
  );
};

export default FilterCheck;