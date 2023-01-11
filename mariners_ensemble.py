import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_ensemble(df, model_list=["log", "forest", "svc"],
                   weights=[0.0, 0.48, 0.52], n_bootstrap=60, stolen_factor=0.97):
    # prepare data for ensemble to examine
    df.dropna(inplace=True)
    # removes identifiers and other undesired variables
    to_remove = ["play_id", "game_id", "catcher_id", "runner_id", "pitcher_id",
                 "batter_id", "c_throw_y", "p_throw_rel_side", "p_throw_rel_height", "p_throw_loc_side",
                 "p_throw_ind_vert_break", "p_throw_spin_rate", "p_throw_spin_axis"]
    handedness = ["pitcher_throws", "batter_side"]
    var_set = [x for x in list(df.columns) if (x not in to_remove)]

    # processes variable set and extracts test set
    X_proc = df[var_set].replace(to_replace=["L", "R"], value=[1, 0]).dropna()
    y_proc = X_proc["is_cs"]
    X_proc, X_test, y_proc, y_test = train_test_split(X_proc.drop(labels="is_cs", axis=1), y_proc, train_size=0.8)
    X_proc["is_cs"] = y_proc

    # performs bootstrapping to increase sample size
    X_train = pd.DataFrame()
    for _ in range(0, n_bootstrap):
        bootstrap_caught = X_proc[X_proc["is_cs"] == 1].sample(frac=0.1)
        k = len(bootstrap_caught)
        bootstrap_stolen = X_proc[X_proc["is_cs"] != 1].sample(n=int(stolen_factor * k))
        X_train = pd.concat([X_train, bootstrap_stolen, bootstrap_caught], ignore_index=True)
    y_train = X_train["is_cs"]
    X_train.drop(labels="is_cs", inplace=True, axis=1)
    scaler = StandardScaler()
    scalar_columns = ~X_train.columns.isin(handedness)
    X_train.loc[:, scalar_columns] = scaler.fit_transform(X_train.loc[:, scalar_columns])
    X_test.loc[:, scalar_columns] = scaler.transform(X_test.loc[:, scalar_columns])
    print("Data cleaned, scaled, and split: size of boostrap training set = {}".format(len(X_train)))

    # checks consistency of weights and number of models to be included
    if len(weights) != len(model_list):
        raise Exception("Dimensional mismatch between weights and models")

    model_dict = {"svc": SVC(), "forest": RandomForestClassifier(max_samples=0.3),
                  "log": LogisticRegression()}

    trained_models = []
    for model_type in model_list:
        trained_models.append(model_dict[model_type].fit(X_train, y_train))
        print("Successfully trained {} for ensemble".format(model_type))

    def ensemble_predict(x):
        indiv_predict = []
        for model in trained_models:
            indiv_predict.append(model.predict(x))
        vote = weights[0]*np.array(indiv_predict[0])
        for i in range(1, len(indiv_predict)):
            vote = vote + weights[i]*np.array(indiv_predict[i])
        vote = vote >= 0.5  # ensemble prediction based on weighted majority vote
        return indiv_predict, vote

    # analyze results
    indiv_pred, y_pred = ensemble_predict(X_test)
    conf_mat_total = confusion_matrix(y_test, y_pred)
    acc_score = np.trace(conf_mat_total) / np.sum(conf_mat_total)
    conf_mat_12 = confusion_matrix(indiv_pred[0], indiv_pred[1])
    conf_mat_13 = confusion_matrix(indiv_pred[0], indiv_pred[2])
    conf_mat_23 = confusion_matrix(indiv_pred[1], indiv_pred[2])
    print("Overall accuracy score: {:.4f}".format(acc_score))

    fig, axs = plt.subplots(2, 2)
    axs[0][0].set_title("Confusion Matrix for Ensemble, x=Predicted, y=True")
    axs[0][1].set_title("Confusion Matrix for y={} vs. x={}".format(model_list[0], model_list[1]))
    axs[1][0].set_title("Confusion Matrix for y={} vs. x={}".format(model_list[0], model_list[2]))
    axs[1][1].set_title("Confusion Matrix for y={} vs. x={}".format(model_list[1], model_list[2]))

    sns.heatmap(conf_mat_total / np.sum(conf_mat_total), annot=True, fmt='.2%', ax=axs[0][0])
    sns.heatmap(conf_mat_12 / np.sum(conf_mat_12), annot=True, fmt='.2%', ax=axs[0][1])
    sns.heatmap(conf_mat_13 / np.sum(conf_mat_13), annot=True, fmt='.2%', ax=axs[1][0])
    sns.heatmap(conf_mat_23 / np.sum(conf_mat_23), annot=True, fmt='.2%', ax=axs[1][1])
    plt.show()


if __name__ == "__main__":
    data = pd.read_csv("mariners_train.csv")
    train_ensemble(data)
