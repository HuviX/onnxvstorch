LABELS = ["negative", "neutral", "positive"]

task = "sentiment"
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"
ONNX_MODEL_PATH = "onnx/twitter-roberta-base-sentiment.onnx"
