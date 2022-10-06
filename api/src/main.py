import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

from app.services import (filter_metric_data_by_metric_id, get_all_metric_data,
                          get_grid_data_per_sampling_strategy)

app = Flask(__name__)
cors = CORS(
    app,
    resources={
        r"/metric_data.json": {"origins": "http://localhost:port"},
        r"/grid_cv_results_model_data.json": {"origins": "http://localhost:port"},
        r"/grid_data_per_sampling_strategy.json": {"origins": "http://localhost:port"},
        r"/grid_best_params.json": {"origins": "http://localhost:port"},
    }
)


@app.route("/", methods=['GET'])
@app.route("/home", methods=['GET'])
@app.route("/directory", methods=['GET'])
def home_directory():
    return jsonify({
        'result': {
            'get_all_metric_data': 'http://localhost:8080/metric_data.json',
            'get_model_data': 'http://localhost:8080/model_data.json',
            'get_dataframe_data': 'http://localhost:8080/dataframe_data.json',
            'get_cross_model_data': 'http://localhost:8080/cross_model_data.json',
            'get_grid_cv_results_model_data': 'http://localhost:8080/grid_cv_results_model_data.json',
            'get_grid_data_per_sampling_strategy': 'http://localhost:8080/grid_data_per_sampling_strategy.json',
            'get_grid_best_params': 'http://localhost:8080/grid_best_params.json',
        }
    })


@app.route("/metric_data.json", methods=['GET'])
@cross_origin(origin='localhost', headers=['Content-Type'])
def get_all_data():
    return jsonify({'result': get_all_metric_data()})


@app.route("/dataframe_data.json", methods=['GET'])
def get_dataframe_data():
    return jsonify({'result': filter_metric_data_by_metric_id(get_all_metric_data(), 'dataframe')})


@app.route("/model_data.json", methods=['GET'])
def get_model_data():
    return jsonify({'result': filter_metric_data_by_metric_id(get_all_metric_data(), 'model')})


@app.route("/cross_model_data.json", methods=['GET'])
def get_cross_model_data():
    return jsonify({'result': filter_metric_data_by_metric_id(get_all_metric_data(), 'cross')})


@app.route("/grid_cv_results_model_data.json", methods=['GET'])
@cross_origin(origin='localhost', headers=['Content-Type'])
def get_grid_cv_results_model_data():
    return jsonify(
        {'result': filter_metric_data_by_metric_id(
            data=get_all_metric_data(), metric_id='grid_cv_results')}
    )


@app.route("/grid_data_per_sampling_strategy.json", methods=['GET'])
@cross_origin(origin='localhost', headers=['Content-Type'])
def grid_data_per_sampling_strategy():
    return jsonify({'result': get_grid_data_per_sampling_strategy()})


@app.route('/grid_best_params.json', methods=['GET'])
@cross_origin(origin='localhost', headers=['Content-Type'])
def get_grid_best_params():
    return jsonify({'result': filter_metric_data_by_metric_id(
        data=get_all_metric_data(), metric_id='grid_best_params')})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
