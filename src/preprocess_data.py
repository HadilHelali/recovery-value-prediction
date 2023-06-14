import os
import sys
import pandas as pd
from sklearn import preprocessing


def count_nulls_by_line(df):
    return df.isnull().sum().sort_values(ascending=False)


def null_percent_by_line(df):
    return (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)


def preprocess_data(data_path):
    df = pd.read_csv(data_path)

    zeros_cnt = count_nulls_by_line(df)
    # df.isnull().sum().sort_values(ascending=False)
    percent_zeros = null_percent_by_line(df)
    # (df.isnull().sum() / df.isnull().count()).sort_values(ascending=False)

    # total number of missing lines and percentage :
    missing_data = pd.concat(
        [zeros_cnt, percent_zeros], axis=1, keys=["Total", "Percent"]
    )

    # getting the indexes of the lines where missing data percentage > 0.15
    drop_list = list(missing_data[missing_data["Percent"] > 0.15].index)

    df.drop(drop_list, axis=1, inplace=True)

    # feature selection
    # TODO : changing it according to our dataset
    df.drop(["Date"], axis=1, inplace=True)
    df.drop(["Location"], axis=1, inplace=True)

    # One-hot encoding of some features
    # TODO : changing it according to our dataset
    ohe = pd.get_dummies(data=df, columns=["WindGustDir", "WindDir9am", "WindDir3pm"])

    # changing the types of some fields
    # TODO : changing it according to our dataset
    ohe["RainToday"] = df["RainToday"].astype(str)
    ohe["RainTomorrow"] = df["RainTomorrow"].astype(str)

    # binary encoding
    # TODO : changing it according to our dataset
    l_b = preprocessing.LabelBinarizer()

    ohe["RainToday"] = l_b.fit_transform(ohe["RainToday"])
    ohe["RainTomorrow"] = l_b.fit_transform(ohe["RainTomorrow"])

    # drop NaN values
    ohe = ohe.fillna(0)
    precessed_df = ohe

    # maybe reordering the columns ?
    cols = precessed_df.columns.tolist()
    cols.remove("RainTomorrow")
    cols.append("RainTomorrow")
    precessed_df = precessed_df[cols]

    features_df = precessed_df.drop(["RainTomorrow"], axis=1)

    # saving the data without the target in a csv file :
    features_df.to_csv("./data/features.csv", index=False)

    # saving the preprocessed data :
    precessed_df.to_csv(data_path[:-4] + "_processed.csv", index=False)


if __name__ == "__main__":
    DATA_PATH = os.path.abspath(sys.argv[1])
    preprocess_data(DATA_PATH)
    print("Saved to {}".format(DATA_PATH[:-4] + "_processed.csv"))
