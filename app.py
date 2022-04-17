from flask import Flask, request
from flask.json import jsonify
import logging

from model import ONNXModel, TorchModel


app = Flask(__name__)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

logging.info("Loading TORCH model ...")
t_model = TorchModel()
logging.info("Model Loaded")

logging.info("Loading ONNX model ...")
o_model = ONNXModel()
logging.info("Model Loaded")


@app.route("/predict_torch", methods=["POST"])
def predict_torch():
    input_data = request.get_json()
    text = input_data["data"]
    resp = t_model.predict(text)
    return jsonify(resp)


@app.route("/predict_onnx", methods=["POST"])
def predict_onnx():
    input_data = request.get_json()
    text = input_data["data"]
    resp = o_model.predict(text)
    return jsonify(resp)
