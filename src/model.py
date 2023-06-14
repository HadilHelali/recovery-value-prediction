import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

"""
Usage : X, y = get_variables(data, target)
* X : the features 
* y : the target
"""


def get_variables(data, column):
    # Seperating the dependant and independant variables
    y = data[column]
    x = data.drop([column], axis=1)

    return x, y


"""
used in train.py 
Usage : pipeline, log_train = model.train(DATA_PATH, PARAM)
* DATA_PATH : the path to the data 
* PARAM : other parameters
"""


def train(data, num_estimators, is_dataframe=False):
    # creating a dataframe from the csv file
    if not is_dataframe:
        data = pd.read_csv(data)

    # TODO:  changing the target to the filed representing the value of the recovery
    # seperating the target from the features :
    x, y = get_variables(data, "RainTomorrow")

    # splitting the train and test set  :
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0
    )

    # defining the pipeline  :
    # TODO : changing the model
    pipe = Pipeline(
        [  # StandardScaler : Standardize features by removing the mean and scaling to unit variance.
            # The standard score of a sample x is calculated as: z = (x - u) / s.
            ("scaler", StandardScaler()),
            (  # Used Logistic Regression
                "LR",
                LogisticRegression(random_state=0, max_iter=num_estimators),
            ),
        ]
    )

    # getting the training logs
    training_logs = pipe.fit(x_train, y_train)

    logs = {"training_logs": training_logs}

    # returning the pipeline and the training logs (Model)
    return pipe, logs


"""
used in evaluate.py 
Usage : log_eval = model.evaluate(DATA_PATH, pipeline, "./results")
* DATA_PATH : the path to the data 
* pipeline : the pipe
* OUTPUT_PATH : "./results"
return : logs : metrics , roc_curve , precision_recall_curve
and saves the plots in /results
"""


def evaluate(data, pipeline, output_path, is_dataframe=False):
    pipe = pipeline

    if not is_dataframe:
        data = pd.read_csv(data)

    # TODO : changing the target field
    y = data["RainTomorrow"]
    x = data.drop(["RainTomorrow"], axis=1)

    # splitting the train and test set  :
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=0
    )

    # metrics
    # TODO : changing the metrics if necessary
    def comb_eval(y_eval, y_pred):
        acc = accuracy_score(y_eval, y_pred)
        recall = recall_score(y_eval, y_pred)
        precision = precision_score(y_eval, y_pred)
        f1 = f1_score(y_eval, y_pred)

        return {"accuracy": acc, "recall": recall, "precision": precision, "f1": f1}

    # y_pred_train = pipe.predict(x_train)
    # train_result = comb_eval(y_train, y_pred_train)

    y_pred_test = pipe.predict(x_test)
    test_result = comb_eval(y_test, y_pred_test)

    # We can perform some cross-validation :
    # cvs = cross_val_score(pipe, x, y, cv=3)

    # roc curve
    # y_pred = pipe.predict(x_test)

    dummy_probs = [0 for _ in range(len(y_test))]
    model_probs = pipe.predict_proba(x_test)
    model_probs = model_probs[:, 1]

    # model_auc = roc_auc_score(y_test, model_probs)

    dummy_fpr, dummy_tpr, _ = roc_curve(y_test, dummy_probs)
    model_fpr, model_tpr, _ = roc_curve(y_test, model_probs)

    # precision_recall_curve
    y_scores = pipe.predict_proba(x_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

    logs = {
        "metrics": test_result,
        "roc_curve": {
            "model_tpr": model_tpr,
            "model_fpr": model_fpr,
            "dummy_tpr": dummy_tpr,
            "dummy_fpr": dummy_fpr,
        },
        "precision_recall_curve": {
            "precisions": precisions,
            "recalls": recalls,
            "thresholds": thresholds,
        },
    }

    # a dummy classifier is a basic, straightforward model that makes predictions using a simple rule or strategy (
    # Random guessing , Most frequent class, Stratified random) It is often used as a baseline or reference point for
    # evaluating the performance of more advanced models in machine learning tasks. The purpose of the dummy
    # classifier is to provide a comparison point for assessing the effectiveness of other models. It serves as a
    # benchmark to understand whether a more complex model is actually learning meaningful patterns in the data or if
    # its performance is no better than random or simple guessing. roc curve plot the roc curve for the model for the
    # dummy classifier
    plt.plot(
        logs["roc_curve"]["dummy_fpr"],
        logs["roc_curve"]["dummy_tpr"],
        linestyle="--",
        label="Dummy Classifier",
    )
    # for our model
    plt.plot(
        logs["roc_curve"]["model_fpr"],
        logs["roc_curve"]["model_tpr"],
        marker=".",
        label="RFC",
    )
    # axis labels
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    # show the legend
    plt.legend()
    out_path = output_path + "/roc_curve.png"
    plt.savefig(out_path, dpi=80)
    plt.cla()

    # Plotting the precisions, recalls and thresholds

    def plot_prc(precisions_plt, recalls_plt, thresholds_plt):
        plt.plot(thresholds_plt, precisions_plt[:-1], "b--", label="Precision")
        plt.plot(thresholds_plt, recalls_plt[:-1], "g-", label="Recall")
        plt.xlabel("Thresholds")
        plt.legend(loc="center left")
        plt.ylim([0, 1])
        out_path_plt = output_path + "/precision_recall_curve.png"
        plt.savefig(out_path_plt, dpi=80)

    # Usage
    plot_prc(
        logs["precision_recall_curve"]["precisions"],
        logs["precision_recall_curve"]["recalls"],
        logs["precision_recall_curve"]["thresholds"],
    )

    return logs
