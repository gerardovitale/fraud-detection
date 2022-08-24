import json

from flask import Flask, jsonify
from flask_cors import CORS, cross_origin

from app.services import get_all_metric_data

app = Flask(__name__)
cors = CORS(app, resources={r"/metric_data.json": {"origins": "http://localhost:port"}})


@app.route("/metric_data.json", methods=['GET'])
@cross_origin(origin='localhost', headers=['Content-Type'])
def get_all_data():
    return jsonify({'result': get_all_metric_data()})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
