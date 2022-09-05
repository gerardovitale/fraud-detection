from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

from app.services import filter_metric_data_by_metric_id, get_all_metric_data

app = Flask(__name__)
cors = CORS(app, resources={
            r"/metric_data.json": {"origins": "http://localhost:port"}})


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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
