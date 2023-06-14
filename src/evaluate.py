import sys
import importlib.util
import os
import json
import joblib

DATA_PATH = os.path.abspath(sys.argv[1])
MODEL_PATH = sys.argv[2]
PICKLE_PATH = sys.argv[3]

sys.path.insert(1, MODEL_PATH)


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = module_from_file("model", MODEL_PATH)

pipeline = joblib.load(PICKLE_PATH)
log_eval = model.evaluate(DATA_PATH, pipeline, "./results")

with open("./results/metrics.json", "w") as outfile:
    json.dump(log_eval["metrics"], outfile)
