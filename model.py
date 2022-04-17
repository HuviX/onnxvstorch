from abc import ABC, abstractmethod

import time
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import torch.nn as nn
import torch

from app_consts import LABELS, MODEL, ONNX_MODEL_PATH
from utils import create_model_for_provider, preprocess, prepare_resp


class Model(ABC):
    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class TorchModel(Model, nn.Module):
    def __init__(self):
        super().__init__()
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    @torch.no_grad()
    def forward(self, x):
        text = preprocess(x)
        encoded_input = self.tokenizer(text, return_tensors="pt")
        start = time.perf_counter()
        output = self.model(**encoded_input)
        finish = time.perf_counter()
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        return scores, finish - start

    def predict(self, x):
        scores, time_spent = self.forward(x)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        resp = {}
        for i in range(scores.shape[0]):
            l = LABELS[ranking[i]]
            s = str(scores[ranking[i]])
            resp[l] = s
        return prepare_resp(resp, time_spent)


class ONNXModel(Model):
    def __init__(self):
        super().__init__()
        self.load_model()

    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.model = create_model_for_provider(ONNX_MODEL_PATH,
                                              "CPUExecutionProvider")

    def forward(self, x):
        text = preprocess(x)
        encoded_input = self.tokenizer(text, return_tensors="pt")
        inputs_onnx = {k: v.numpy() for k, v in encoded_input.items()}
        start = time.perf_counter()
        output = self.model.run(None, inputs_onnx)
        finish = time.perf_counter()
        scores = output[0][0]
        scores = softmax(scores)
        return scores, finish - start

    def predict(self, x):
        scores, time_spent = self.forward(x)
        ranking = np.argsort(scores)
        ranking = ranking[::-1]
        resp = {}
        for i in range(scores.shape[0]):
            l = LABELS[ranking[i]]
            s = str(scores[ranking[i]])
            resp[l] = s
        return prepare_resp(resp, time_spent)
