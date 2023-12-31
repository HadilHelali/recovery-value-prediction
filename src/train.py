import sys
import importlib.util
import os
import joblib

# import time

# Getting the paths from the command line ?
DATA_PATH = os.path.abspath(sys.argv[1])
# PROJ_PATH = os.path.abspath(sys.argv[2])
MODEL_PATH = sys.argv[2]
PARAM = int(sys.argv[3])

sys.path.insert(1, MODEL_PATH)


# Getting the model from the output file :


def module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


model = module_from_file("model", MODEL_PATH)

if __name__ == "__main__":
    # training the model using the train function from model.py
    pipeline, log_train = model.train(DATA_PATH, PARAM)

    # if sys.argv[4]:
    # with open("./models/model.pkl", "wb") as file:
    #     pickle.dump(pipeline[0], file)

    # TODO : modifications : adding the time of the model's creation as the name
    # TODO : uncomment the time import
    # ts = int(time.time())
    # joblib.dump(pipeline, "./models/"+ str(ts)+".joblib")

    # saving the model after train :
    joblib.dump(pipeline, "./models/model.joblib")
